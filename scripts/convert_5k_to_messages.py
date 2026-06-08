import json
SYSTEM = 'You are an expert reasoning model. Solve puzzles step by step, showing all your work. Place your final answer inside \\boxed{}.'
examples = []
with open('/home/a474r867/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        examples.append({'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': d['prompt'] + '\n\nThink step by step and place your final answer inside \\boxed{}.'},
            {'role': 'assistant', 'content': d['answer']}
        ]})
with open('/home/a474r867/scratch/nemotron-eco-reasoner/data/kaggle_5k_messages.jsonl', 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex) + '\n')
print('Converted %d examples' % len(examples))
