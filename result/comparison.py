import json

# Load the baseline predictions (50 samples you analyzed)
baseline_samples = []
with open('misclassified_samples_baseline.jsonl', 'r') as f:
    for line in f:
        baseline_samples.append(json.loads(line))

# Load the stage2 predictions
with open('vqa_rad_stage2_hybrid_predictions.json', 'r') as f:
    stage2_data = json.load(f)

# Create a mapping of encounter_id to stage2 predictions
stage2_predictions = {p['encounter_id']: p for p in stage2_data['predictions']}

# Analyze the same 50 samples in stage2
comparison_results = []

for baseline_sample in baseline_samples:
    encounter_id = baseline_sample['encounter_id']
    
    if encounter_id in stage2_predictions:
        stage2_sample = stage2_predictions[encounter_id]
        
        comparison = {
            'encounter_id': encounter_id,
            'question': baseline_sample['question'],
            'gold': baseline_sample['gold'],
            'baseline_prediction': baseline_sample['prediction'],
            'baseline_match': baseline_sample['match'],
            'stage2_prediction': stage2_sample['prediction'],
            'stage2_match': stage2_sample['match'],
            'improvement': None,  # Will calculate: 'better', 'worse', 'same'
        }
        
        # Determine improvement
        if baseline_sample['match'] and not stage2_sample['match']:
            comparison['improvement'] = 'worse'
        elif not baseline_sample['match'] and stage2_sample['match']:
            comparison['improvement'] = 'better'
        else:
            comparison['improvement'] = 'same'
            
        comparison_results.append(comparison)

# Save comparison results
with open('stage2_comparison_analysis.jsonl', 'w') as f:
    for result in comparison_results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

# Calculate statistics
total_samples = len(comparison_results)
improved = sum(1 for r in comparison_results if r['improvement'] == 'better')
worsened = sum(1 for r in comparison_results if r['improvement'] == 'worse')
unchanged = sum(1 for r in comparison_results if r['improvement'] == 'same')

print(f"Stage 2 vs Baseline Comparison Analysis")
print("=" * 50)
print(f"Total samples analyzed: {total_samples}")
print(f"Improved predictions: {improved} ({improved/total_samples*100:.1f}%)")
print(f"Worsened predictions: {worsened} ({worsened/total_samples*100:.1f}%)")
print(f"Unchanged predictions: {unchanged} ({unchanged/total_samples*100:.1f}%)")
print(f"Net improvement: {improved - worsened} samples")

# Show examples of improvements
print(f"\nExamples of improved predictions:")
for result in comparison_results[:5]:  # Show first 5 improvements
    if result['improvement'] == 'better':
        print(f"  {result['encounter_id']}:")
        print(f"    Question: {result['question']}")
        print(f"    Gold: {result['gold']}")
        print(f"    Baseline: {result['baseline_prediction']} ❌")
        print(f"    Stage 2: {result['stage2_prediction']} ✅")

print(f"\nGenerated stage2_comparison_analysis.jsonl")