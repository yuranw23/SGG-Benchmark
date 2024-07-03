import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    output_dir = f'/home/yuranw/SGG-Benchmark/output/{args.model}/'
    print(output_dir)

    # result_dict = torch.load('inference/result_dict.pytorch')
    eval_results = torch.load(f'{output_dir}inference/eval_results.pytorch')
    groundtruths = eval_results['groundtruths']
    predictions = eval_results['predictions']

    # metrics calculation
    correct_obj = 0
    total_obj = 0

    for prediction in predictions:
        labels = prediction.get_field('labels')
        predict_logits = prediction.get_field('predict_logits')
        pred_labels = torch.argmax(predict_logits, axis=1)
        
        # pred_rel_labels = prediction.get_field('pred_rel_labels') # [29, 29] relation 
        # rel_pair_idxs = prediction.get_field('rel_pair_idxs') # [[2,3], [2,4]] every pair of objects
        # labels, predict_logits, pred_labels, pred_scores, rel_pair_idxs, pred_rel_scores, pred_rel_labels
        
        # pred_rel_scores = prediction.get_field('pred_rel_scores') # (20 combination, 51: 50 rels + none)
        
        # obj metric
        correct_obj += sum(labels == pred_labels).item()
        total_obj += len(labels)
    
    print(f'obj acc = {correct_obj / total_obj}')