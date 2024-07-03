import torch
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    output_dir = f'/home/yuranw/SGG-Benchmark/output/{args.model}/'
    print(output_dir)

    # result_dict = torch.load('inference/result_dict.pytorch')
    vocab_file = json.load(open('/home/yuranw/SGG-Benchmark/datasets/VG150/VG-SGG-dicts-with-attri.json'))
    idx2label = vocab_file['idx_to_label']
    idx2pred = vocab_file['idx_to_predicate']

    eval_results = torch.load(f'{output_dir}inference/eval_results.pytorch')
    groundtruths = eval_results['groundtruths']
    predictions = eval_results['predictions']

    # metrics calculation
    correct_obj = 0
    total_obj = 0
    correct_pred = 0
    total_pred = 0

    correct_pred_dict = {}
    total_pred_dict = {}

    for i in range(len(predictions)):
        groundtruth = groundtruths[i]
        prediction = predictions[i]
        
        # obj metric
        labels = prediction.get_field('labels')
        predict_logits = prediction.get_field('predict_logits')
        pred_labels = torch.argmax(predict_logits, axis=1)
        correct_obj += sum(labels == pred_labels).item()
        total_obj += len(labels)

        # predicate metric
        gt_rels = groundtruth.get_field('relation_tuple')
        total_pred += len(gt_rels)
        pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
        pred_rel_label = prediction.get_field('pred_rel_scores')
        pred_rel_label[:,0] = 0
        pred_rel_score, pred_rel_label = pred_rel_label.max(-1)

        for gt_rel in gt_rels:
            obj_pair = gt_rel[:2].tolist()
            gt_pred = gt_rel[-1].item()
            pair_idx = pred_rel_pair.index(obj_pair)
            pred_rel = pred_rel_label[pair_idx].item()

            gt_pred_eng = idx2pred[str(gt_pred)]
            if gt_pred_eng not in total_pred_dict:
                total_pred_dict[gt_pred_eng] = 0
                correct_pred_dict[gt_pred_eng] = 0
            if pred_rel == gt_pred:
                correct_pred += 1
                correct_pred_dict[gt_pred_eng] += 1
                total_pred_dict[gt_pred_eng] += 1
            else:
                total_pred_dict[gt_pred_eng] += 1

             
        
    print(f'correct obj = {correct_obj} || total obj = {total_obj} || correct pred = {correct_pred} || total pred = {total_pred}')
    print(f'obj acc = {correct_obj / total_obj}')
    print(f'pred acc = {correct_pred / total_pred}')

    macc_dict = {}
    for pred in correct_pred_dict:
        macc_dict[pred] = correct_pred_dict[pred] / total_pred_dict
    
    print(macc_dict)
    print("=========================")
    print(correct_pred_dict)
    print("=========================")
    print(total_pred_dict)