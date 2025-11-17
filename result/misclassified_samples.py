import json
import random

with open('vqa_rad_stage2_hybrid_predictions.json', 'r') as f:
    data = json.load(f)

corrected_ids = set()
try:
    with open('corrected_cases_hybrid.jsonl', 'r') as f:
        for line in f:
            corrected_case = json.loads(line)
            corrected_ids.add(corrected_case['encounter_id'])
except FileNotFoundError:
    print("corrected_cases_hybrid.jsonl not found, proceeding without exclusion")

misclassified_samples = [item for item in data['predictions'] if not item['match'] and item['encounter_id'] not in corrected_ids]

print(f"Total misclassified samples: {len(misclassified_samples)}")
print(f"Excluded corrected cases: {len(corrected_ids)}")

if len(misclassified_samples) >= 50:
    selected_samples = random.sample(misclassified_samples, 50)
else:
    selected_samples = misclassified_samples

output_filename = 'misclassified_samples.jsonl'
with open(output_filename, 'w') as f:
    for sample in selected_samples:
        f.write(json.dumps(sample) + '\n')

print(f"Selected {len(selected_samples)} misclassified samples (excluding corrected cases)")
print(f"Saved to {output_filename}")