import yaml
from src.fetcher   import SECFetcher
from src.parser    import SECParser
from src.chunker   import DocumentChunker
from src.generator import QAGenerator
from src.verifier  import AnswerVerifier

with open("config.yaml") as f:
    config = yaml.safe_load(f)

doc      = SECFetcher(config).fetch()
sections = SECParser(config).parse(doc.raw_html)
chunks   = DocumentChunker(config).chunk_sections(sections)

# Generate 8 pairs for testing
generator = QAGenerator(config)
pairs     = generator.generate_from_chunks(chunks[:8], target=8)

print(f"\nGenerated {len(pairs)} pairs — now verifying...\n")

verifier = AnswerVerifier(config, chunks)
verified = verifier.verify_all(pairs)

print(f"\n{'='*55}")
print(f"VERIFIED: {len(verified)} / {len(pairs)}")
print(f"{'='*55}")
for v in verified:
    print(f"\nQ    : {v.question}")
    print(f"TYPE : {v.question_type} | DIFF: {v.difficulty}")
    print(f"CHECK: match={v.verification.source_match_score:.2f} | "
          f"verdict={v.verification.llm_verdict} | "
          f"conf={v.verification.confidence}")