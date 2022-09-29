from itertools import combinations

import torch
from rouge import rouge_score
from torch import nn
from transformers import BartForConditionalGeneration
from bert_score import BERTScorer


def fast_rouge(summary_batch, output_batch, device):
    rouge_scores = []
    for bs in range(len(summary_batch)):
        ref = summary_batch[bs]
        sys = output_batch[bs]
        rouge_scores.append(rouge_score.rouge_n(sys, ref, 1)['f'] + rouge_score.rouge_n(sys, ref, 2)['f'])
    return torch.tensor(rouge_scores).to(device)


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, mask_cls):
        h = self.linear(inputs).squeeze(-1)  # [batch_size, seq_len]
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class BaselineExtModel(nn.Module):
    def __init__(self, args, article_dict=None):
        super(BaselineExtModel, self).__init__()
        full_bart = BartForConditionalGeneration.from_pretrained(f"facebook/bart-{args.version}")
        full_bart.resize_token_embeddings(full_bart.config.vocab_size + 3)
        self.encoder = full_bart.get_encoder()
        self.hidden_size = self.encoder.config.hidden_size
        self.decoder = Classifier(self.hidden_size)
        self.hidden_size = self.encoder.config.hidden_size

        self.metric = args.metric
        self.block_trigram = args.block_trigram
        self.pad_id = args.pad_id
        self.dataset = args.dataset
        self.ext_num = args.ext_num

        self.article_dict = article_dict

    def get_output(self, pred, mask, article_id_int):
        batch_size = pred.size(0)
        pred = pred + 1e3 * mask.float()
        ext_ids = torch.sort(pred, dim=-1, descending=True).indices
        output_batch = []
        summary_batch = []
        for bs in range(batch_size):
            art_id = str(article_id_int[bs].item())
            text = self.article_dict[art_id]["text"]

            ext_summs = []
            for sent_id in ext_ids[bs]:
                if sent_id >= len(text) or len(ext_summs) >= self.ext_num:
                    break
                ext_summs.append(text[sent_id])
            output_batch.append(ext_summs)
            summary_batch.append(self.article_dict[art_id]["summary"])

        return output_batch, summary_batch

    def forward(self, text_id, cls_ids, article_id_int):

        batch_size = text_id.size(0)
        doc_inp_mask = ~(text_id == self.pad_id)
        mask_cls = 1 - (cls_ids == -1).long()
        bert_out = self.encoder(text_id, attention_mask=doc_inp_mask)[0]  # last layer
        sent_emb = bert_out[torch.arange(batch_size).unsqueeze(1), cls_ids]
        sent_emb = sent_emb * mask_cls.unsqueeze(-1).float()
        assert sent_emb.size() == (batch_size, cls_ids.size(1), self.hidden_size)  # [batch_size, seq_len, hidden_size]

        sent_scores = self.decoder(sent_emb, mask_cls)  # [batch_size, seq_len]
        assert sent_scores.size() == (text_id.size(0), cls_ids.size(1))

        # produce validation score
        if self.training:
            return {'pred': sent_scores, 'mask': mask_cls}
        else:
            output_batch, summary_batch = self.get_output(sent_scores, mask_cls, article_id_int)
            return {'gold_ref': summary_batch, "output": output_batch, "article_id": article_id_int,
                    "rouge": fast_rouge(summary_batch, output_batch, sent_scores.device)}


def check_n_gram(sentences, n):
    all_sents = sentences[0]
    for sentence in sentences[1:]:
        tokens = sentence.split(' ')
        s_len = len(tokens)
        for i in range(s_len):
            if i + n > s_len:
                break
            if ' '.join(tokens[i: i + n]) in all_sents:
                return False
        all_sents = all_sents + " " + sentence
    return True  # no n_gram overlap


class CoLoExtModel(nn.Module):
    def __init__(self, args, article_dict=None):
        super(CoLoExtModel, self).__init__()
        full_bart = BartForConditionalGeneration.from_pretrained(f"facebook/bart-{args.version}")
        full_bart.resize_token_embeddings(full_bart.config.vocab_size + 3)
        self.encoder = full_bart.get_encoder()
        self.hidden_size = self.encoder.config.hidden_size
        self.decoder = Classifier(self.hidden_size)

        self.pad_id = args.pad_id
        self.dataset = args.dataset
        self.ext_num = args.ext_num  # is used to clip the original size of the the document
        self.metric = args.metric
        self.block_trigram = args.block_trigram
        #### warning: using bert_score will result in long training time ####
        if self.metric == "bert_score":
            self.scorer = BERTScorer(model_type="bert-base-uncased", rescale_with_baseline=True, lang='en')

        self.article_dict = article_dict

    def generate_sorted_embedding(self, article_id_int, sent_indices, sent_embs):
        article_id = str(article_id_int.item())
        sents = [self.article_dict[article_id]['text'][i] for i in sent_indices]
        summary = self.article_dict[article_id]['summary']
        sent_id = list(range(min(len(sents), self.ext_num)))

        if "CNNDM" in self.dataset:
            possible_sent_num1 = 2
            possible_sent_num2 = 3
            indices = list(combinations(sent_id, possible_sent_num1))
            indices += list(combinations(sent_id, possible_sent_num2))
            if self.training:
                MAX_CAND_NUM = 50
                indices = indices[:MAX_CAND_NUM]
        else:
            raise Exception("you can add the config  for your dataset here ")
        candidate_strings = []
        candidate_embs = []
        for index_pair in indices:
            candidate_strings.append([sents[index_single] for index_single in index_pair])
            selected_emb = sent_embs.index_select(0, torch.tensor(index_pair, device=sent_embs.device))
            candidate_embs.append(torch.mean(selected_emb, dim=0))
        # inference mode , do not sort and return candidate strings
        if not self.training:
            return torch.stack(candidate_embs), candidate_strings, summary

        # training mode, return sorted embedding
        scores = []

        for candidate_str in candidate_strings:
            if self.metric == "rouge":
                scores.append(rouge_score.rouge_n(summary, candidate_str, 1)['f'] +
                              rouge_score.rouge_n(summary, candidate_str, 2)['f'])
            elif self.metric == "bert_score":
                summary = [" ".join(summary)]
                candidate_str = [" ".join(candidate_str)]
                scores.append(self.scorer.score(candidate_str, summary)[2][0])
            else:
                raise Exception("you can add the scoring function here ")

        # 将每个 batch 的 candidate_emb 按照 rouge 排序
        sorted_scores = sorted(zip(candidate_embs, scores), key=lambda x: x[1], reverse=True)
        sorted_embs = torch.stack([x[0] for x in sorted_scores])
        return sorted_embs

    def forward(self, text_id, cls_ids, article_id_int):
        batch_size = text_id.size(0)
        doc_inp_mask = ~(text_id == self.pad_id)
        mask_cls = 1 - (cls_ids == -1).long()
        bart_out = self.encoder(text_id, attention_mask=doc_inp_mask)[0]  # last layer  batch x seq_len x h
        sent_emb = bart_out[torch.arange(batch_size).unsqueeze(1), cls_ids]
        sent_emb = sent_emb * mask_cls.unsqueeze(-1).float()
        assert sent_emb.size() == (batch_size, cls_ids.size(1), self.hidden_size)  # [batch_size, seq_len, hidden_size]

        doc_emb = bart_out[:, 0, :]
        sent_scores = self.decoder(sent_emb, mask_cls)  # [batch_size, seq_len]
        assert sent_scores.size() == (text_id.size(0), cls_ids.size(1))
        pred = sent_scores + mask_cls.float()
        ext_ids = torch.topk(pred, k=min(self.ext_num, cls_ids.size(1)), dim=-1, largest=True).indices
        selected_sent_embs = sent_emb[torch.arange(batch_size).unsqueeze(1), ext_ids]
        if self.training:
            # sort the candidate embedings according to scoring function
            sorted_embs = []
            for bs in range(batch_size):
                sorted_embs.append(
                    self.generate_sorted_embedding(article_id_int[bs], ext_ids[bs], selected_sent_embs[bs]))
            sorted_embs = torch.stack(sorted_embs, dim=0)
            doc_emb = doc_emb.unsqueeze(1).expand_as(sorted_embs)
            score = torch.cosine_similarity(sorted_embs, doc_emb, dim=-1)  # [batch_size, candidate_num]
            return {'pred': sent_scores, 'mask': mask_cls, 'score': score}
        else:
            candidate_emb_batch = []  # candidate
            candidate_string_batch = []  # string
            summary_batch = []
            for bs in range(batch_size):
                candidate_emb, candidate_strings, summary = self.generate_sorted_embedding(article_id_int[bs],
                                                                                           ext_ids[bs],
                                                                                           selected_sent_embs[bs])
                candidate_emb_batch.append(candidate_emb)
                candidate_string_batch.append(candidate_strings)
                summary_batch.append(summary)
            candidate_emb_batch = torch.stack(candidate_emb_batch)
            doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb_batch)
            score = torch.cosine_similarity(candidate_emb_batch, doc_emb, dim=-1)  # [batch, candidate_num]
            max_ids = torch.topk(score, k=min(3, score.size(-1)), largest=True, dim=-1).indices
            output_batch = []
            for bs in range(batch_size):
                output_batch.append(candidate_string_batch[bs][max_ids[bs][0]])
            if self.block_trigram:
                # trigram blocking
                for bs in range(batch_size):
                    candidate = output_batch[bs]
                    for max_id in max_ids[bs]:
                        if check_n_gram(candidate_string_batch[bs][max_id], 3):
                            candidate = candidate_string_batch[bs][max_id]
                            break
                    output_batch[bs] = candidate
            return {'gold_ref': summary_batch, "output": output_batch,
                    "rouge": fast_rouge(summary_batch, output_batch, score.device)}
