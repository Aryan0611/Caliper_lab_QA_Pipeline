"""
Test the full pipeline with a small batch
Run: python test_pipeline.py
"""

import yaml
from src.pipeline import QAPipeline

print("=" * 60)
print("PIPELINE VERIFICATION TEST")
print("=" * 60)

# Run pipeline with small target
pipeline = QAPipeline(config_path="config.yaml", debug=True)
results = pipeline.run(target_pairs=15)  # Small batch for testing

# Check results
print("\n" + "=" * 60)
print("VERIFICATION CHECKS")
print("=" * 60)

checks_passed = 0
checks_total = 0

# Check 1: Pipeline succeeded
checks_total += 1
if results.get("success"):
    print("✅ Check 1: Pipeline completed successfully")
    checks_passed += 1
else:
    print(f"❌ Check 1: Pipeline failed - {results.get('error')}")

# Check 2: Generated pairs
checks_total += 1
generated = results.get("total_generated", 0)
if generated >= 10:
    print(f"✅ Check 2: Generated {generated} pairs (target was 15)")
    checks_passed += 1
else:
    print(f"❌ Check 2: Only generated {generated} pairs")

# Check 3: Verified pairs
checks_total += 1
verified = results.get("total_verified", 0)
if verified >= 5:
    print(f"✅ Check 3: Verified {verified} pairs")
    checks_passed += 1
else:
    print(f"❌ Check 3: Only {verified} pairs verified")

# Check 4: Pass rate reasonable
checks_total += 1
pass_rate = results.get("pass_rate", 0)
if pass_rate >= 0.5:  # At least 50% pass rate
    print(f"✅ Check 4: Pass rate {pass_rate:.1%} is acceptable")
    checks_passed += 1
else:
    print(f"❌ Check 4: Pass rate {pass_rate:.1%} is too low")

# Check 5: Output files exist
checks_total += 1
from pathlib import Path
json_exists = Path("output/qa_pairs.json").exists()
csv_exists = Path("output/qa_pairs.csv").exists()
if json_exists and csv_exists:
    print("✅ Check 5: Output files created (JSON + CSV)")
    checks_passed += 1
else:
    print(f"❌ Check 5: Missing output files (JSON={json_exists}, CSV={csv_exists})")

# Check 6: JSON is valid and has correct structure
checks_total += 1
try:
    import json
    with open("output/qa_pairs.json") as f:
        data = json.load(f)
    
    has_metadata = "metadata" in data
    has_qa_pairs = "qa_pairs" in data
    has_stats = "statistics" in data
    
    if has_metadata and has_qa_pairs and has_stats:
        print("✅ Check 6: JSON structure is valid")
        checks_passed += 1
    else:
        print(f"❌ Check 6: JSON missing fields")
except Exception as e:
    print(f"❌ Check 6: JSON parse error - {e}")

# Check 7: Q&A pairs have required fields
checks_total += 1
try:
    qa_pairs = data.get("qa_pairs", [])
    if qa_pairs:
        sample = qa_pairs[0]
        required_fields = ["id", "question", "ground_truth_answer", 
                          "source_passage", "question_type", "difficulty"]
        missing = [f for f in required_fields if f not in sample]
        
        if not missing:
            print("✅ Check 7: Q&A pairs have all required fields")
            checks_passed += 1
        else:
            print(f"❌ Check 7: Missing fields: {missing}")
    else:
        print("❌ Check 7: No Q&A pairs in output")
except Exception as e:
    print(f"❌ Check 7: Error checking fields - {e}")

# Check 8: CSV is readable
checks_total += 1
try:
    import pandas as pd
    df = pd.read_csv("output/qa_pairs.csv")
    if len(df) > 0:
        print(f"✅ Check 8: CSV readable with {len(df)} rows")
        checks_passed += 1
    else:
        print("❌ Check 8: CSV is empty")
except Exception as e:
    print(f"❌ Check 8: CSV read error - {e}")

# Final summary
print("\n" + "=" * 60)
print(f"VERIFICATION RESULT: {checks_passed}/{checks_total} checks passed")
print("=" * 60)

if checks_passed == checks_total:
    print("\n🎉 ALL CHECKS PASSED! Pipeline is working correctly.")
else:
    print(f"\n⚠️  {checks_total - checks_passed} checks failed. Review issues above.")

# Show sample output
print("\n" + "=" * 60)
print("SAMPLE OUTPUT")
print("=" * 60)

verified_pairs = pipeline.get_verified_pairs()
if verified_pairs:
    for i, pair in enumerate(verified_pairs[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(f"ID: {pair.id}")
        print(f"Type: {pair.question_type} | Difficulty: {pair.difficulty}")
        print(f"Q: {pair.question[:80]}...")
        print(f"A: {pair.ground_truth_answer[:100]}...")
        print(f"Verification: {pair.verification.status} (score={pair.verification.source_match_score:.2f})")