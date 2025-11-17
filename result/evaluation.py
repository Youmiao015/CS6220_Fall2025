import json

def longest_common_word_ratio(str1, str2):
    """Calculate the ratio of common words to gold words"""
    str1 = str(str1).lower().strip()
    str2 = str(str2).lower().strip()
    
    if not str1:
        return 0.0
    
    # Split into words
    words1 = str1.split()
    words2 = str2.split()
    
    if not words1:
        return 0.0
    
    # Find all common words
    common_words = set(words1) & set(words2)
    
    # Return common words count / gold words count
    return len(common_words) / len(words1)

# Load data
with open('vqa_rad_stage2_hybrid_predictions.json', 'r') as f:
    data = json.load(f)

# Use single threshold 0.5
threshold = 0.5

print(f"Threshold {threshold} correction analysis")
print("=" * 50)

corrected_count = 0

# Save corrected cases to corrected_cases.jsonl
with open('corrected_cases.jsonl', 'w') as f_corrected:
    for p in data['predictions']:
        similarity = longest_common_word_ratio(p['gold'], p['prediction'])
        new_match = similarity >= threshold
        
        # Corrected cases: originally wrong, now correct
        if not p['match'] and new_match:
            corrected_case = {
                'encounter_id': p['encounter_id'],
                'question': p['question'],
                'gold': p['gold'],
                'prediction': p['prediction'],
                'similarity': round(similarity, 3),
                'threshold': threshold
            }
            f_corrected.write(json.dumps(corrected_case, ensure_ascii=False) + '\n')
            corrected_count += 1

print(f"Corrected cases: {corrected_count}")
print(f"Generated corrected_cases.jsonl")