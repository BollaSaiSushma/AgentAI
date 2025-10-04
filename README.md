Fixed-size Chunking






Advantages

Simplicity – Very easy to implement (just split text into N tokens or characters).

Consistency – All chunks are roughly the same size, which makes embedding storage and retrieval predictable.

Performance – Faster indexing since chunks don’t need NLP parsing or semantic grouping.

Balanced retrieval – Works well when you don’t care about natural sentence/paragraph breaks but want uniform context for embeddings.

Example

Suppose we have a passage:

"Artificial Intelligence is transforming industries worldwide. 
It is used in healthcare for diagnostics and in finance for fraud detection. 
Self-driving cars also rely on AI to navigate safely."


If we apply Fixed-size Chunking with a chunk size = 10 words, we get:

Chunk 1: "Artificial Intelligence is transforming industries worldwide. It is"
Chunk 2: "used in healthcare for diagnostics and in finance for"
Chunk 3: "fraud detection. Self-driving cars also rely on AI"
Chunk 4: "to navigate safely."


Each chunk has ~10 words, regardless of sentence boundaries.

Easy to generate embeddings for each chunk.

When to use

For long documents where sentence/paragraph boundaries don’t matter much.

When you want fast, uniform chunking (e.g., building a vector store quickly).

In cases where LLM’s context window is small and you want evenly sized retrieval units.




Splits text into uniform token/word/character blocks.
✅ Use when:

You need simplicity and speed.

Documents are unstructured (emails, scraped web text).

Your LLM has a small context window (4k–8k tokens).
⚠️ Risk: Can break semantic meaning.

Example: Splitting logs or transcripts into 512-token chunks for embedding.





Overlapping Chunking







📌 What it is

Instead of splitting text into non-overlapping fixed-size chunks, we allow some overlap (shared content) between consecutive chunks.

This ensures context continuity — important because LLMs may miss meaning if a sentence or concept is split between chunks.

Think of it as a sliding window over the text.

✅ Advantages

Preserves context → If a sentence or paragraph crosses chunk boundaries, the overlap ensures it’s still captured.

Improves retrieval quality → Embeddings overlap, so relevant passages are less likely to be “cut off.”

Better for Q&A → User questions often relate to concepts spanning multiple sentences. Overlap increases the chance that all relevant info is in at least one chunk.

Balances size and recall → You avoid excessively large chunks while still maintaining coherence.

📘 Example

Text:

"Artificial Intelligence is transforming industries worldwide. 
It is used in healthcare for diagnostics and in finance for fraud detection. 
Self-driving cars also rely on AI to navigate safely."


Chunk size = 12 words

Overlap = 4 words

Result:

Chunk 1: "Artificial Intelligence is transforming industries worldwide. It is used"
Chunk 2: "is used in healthcare for diagnostics and in finance for fraud"
Chunk 3: "finance for fraud detection. Self-driving cars also rely on AI"
Chunk 4: "also rely on AI to navigate safely."


👉 Notice:

The words "is used" appear in both Chunk 1 and Chunk 2.

"finance for fraud" appears in both Chunk 2 and Chunk 3.

This overlap ensures that even if the retrieval system grabs only one chunk, the context isn’t lost.

🚀 When to use

In RAG pipelines where precision and context continuity are critical (chatbots, legal/medical docs).

When chunks are small (e.g., 256–512 tokens) but you want context that flows across them.

When working with QA systems where questions may depend on adjacent sentences.

📌 Summary:

Overlapping Chunking = Fixed-size Chunking + Context Sharing.

It trades off a little more storage/compute for much higher retrieval accuracy.



Fixed-size chunks but with overlaps for continuity.
✅ Use when:

Queries often depend on multi-sentence context.

Your model struggles with cut-off sentences.

You’re using embeddings for semantic search in RAG.
⚠️ Trade-off: More storage/compute.

Example: QA assistant where a law clause or medical explanation might be split mid-sentence. Overlap ensures both chunks cover the boundary.








Semantic Chunking







📌 What it is

Instead of splitting by size (tokens/characters) or structure (sentences/paragraphs), Semantic Chunking uses meaning to decide where to cut.

Typically done with embeddings, topic modeling, or similarity clustering.

Ensures each chunk captures a complete idea or topic.

✅ Advantages

Meaning-preserving → Chunks represent complete concepts instead of arbitrary cuts.

Improves retrieval accuracy → Since chunks are contextually coherent, search results align better with user queries.

Flexible size → Chunks adapt naturally to topic boundaries (some short, some longer).

Less redundancy → Unlike overlapping chunking, semantic chunking minimizes duplication.

Great for long documents → Splits large reports, manuals, or research papers into meaningful sections.

📘 Example

Text:

"AI is widely used in healthcare. It helps with diagnostics and patient data analysis. 
In finance, AI is applied for fraud detection, algorithmic trading, and risk management. 
Meanwhile, self-driving cars rely on AI for navigation and safety."

Fixed-size Chunking (10 words each)
Chunk 1: "AI is widely used in healthcare. It helps with"
Chunk 2: "diagnostics and patient data analysis. In finance, AI"
Chunk 3: "is applied for fraud detection, algorithmic trading, and"
Chunk 4: "risk management. Meanwhile, self-driving cars rely on"
Chunk 5: "AI for navigation and safety."


⚠️ Problem: Cuts across sentences/ideas.




Splits text by meaning using embeddings/clustering.
✅ Use when:

Documents are dense and knowledge-heavy (legal, research, enterprise docs).

You need topic-coherent chunks for precise retrieval.

You want to minimize overlap while keeping meaningful units.
⚠️ Risk: More compute, harder to implement.

Example: Splitting a medical textbook into chunks about diagnosis, treatment, and patient outcomes.






Regex / Rule-based Chunking











📌 What it is

Instead of splitting by size or semantics, you define rules or regular expressions (regex) to split text at specific patterns (like Q&A, bullet points, code blocks, timestamps, etc.).

Works best when documents follow a structured or semi-structured format.

Example: FAQs, chat logs, JSON logs, CSVs, meeting transcripts.

✅ Advantages

Precise control → You decide exactly where splits happen (e.g., at "Q:", "###", or "ERROR:").

Preserves structure → Maintains document formatting (headings, lists, dialogues).

Efficient for repetitive patterns → Works perfectly for FAQs, policies, transcripts, or log files.

Reduces noise → Keeps chunks aligned with meaningful units of text.

Domain adaptable → You can design regex to match industry-specific formats (e.g., ICD codes in healthcare, timestamps in transcripts).

📘 Example
Original Text (FAQ format):
Q: What is AI?
A: Artificial Intelligence is the simulation of human intelligence by machines.

Q: Where is AI used?
A: It is applied in healthcare, finance, and transportation.

Q: What is Machine Learning?
A: ML is a subset of AI focused on pattern recognition and prediction.

Rule/Regex Pattern:

Split whenever "Q:" appears.

Resulting Chunks:
Chunk 1: "Q: What is AI? A: Artificial Intelligence is the simulation of human intelligence by machines."
Chunk 2: "Q: Where is AI used? A: It is applied in healthcare, finance, and transportation."
Chunk 3: "Q: What is Machine Learning? A: ML is a subset of AI focused on pattern recognition and prediction."


👉 Now each chunk is a full Q&A pair — perfect for RAG retrieval.

🚀 When to use

FAQs & Knowledge Bases (Q&A patterns).

Transcripts (timestamps like [00:01:23]).

Code/documentation (split by functions, classes, ### headers).

Logs (split by ERROR, WARN, or JSON delimiters).

📌 Summary:

Regex / Rule-based chunking gives you precision when your data has clear patterns.

It avoids arbitrary cuts and keeps domain-specific structure intact.

Best when documents have a consistent format.

👉 Would you like me to also create a Python code snippet showing how to implement regex-based chunking for FAQs or log files?




Splits first into big sections (chapters, headings), then into smaller sub-chunks.
✅ Use when:

Documents have clear structure (manuals, legal docs, books, policies).

You want multi-level retrieval (high-level + detailed).

You need to preserve hierarchy for citations.
⚠️ Risk: Requires structured input & parsing.

Example: A software manual where retrieval may target either a chapter title or a specific step inside it.





Hierarchical Chunking




📌 What it is

Hierarchical Chunking means splitting text in multiple levels:

First level → big sections (chapters, headings, topics).

Second level → within each section, break into smaller sub-chunks (paragraphs, sentences, or token blocks).

This way, you maintain the document’s structure and ensure both high-level context and fine-grained details are preserved.

✅ Advantages

Preserves document hierarchy

Captures the natural structure of manuals, books, legal docs, or technical specs.

Keeps context of “where” each chunk belongs.

Multi-scale retrieval

Retrieval systems can choose large chunks for broad context or small chunks for detailed answers.

Improves relevance

Since chunks are tied to section titles (e.g., “Chapter 3 – Safety Guidelines”), retrieval aligns better with queries.

Efficient for long documents

Instead of only flat small chunks (which may scatter meaning), hierarchical keeps related info grouped.

Supports citation and navigation

You can trace a chunk back to its parent section, making citations and UI presentation clearer.


Splits by patterns (timestamps, bullet points, headers, Q&A).
✅ Use when:

Text is highly structured (logs, FAQs, transcripts, JSON).

You want precise control over splits.

Your domain has repetitive formats.
⚠️ Risk: Brittle, needs custom rules per domain.

Example: Splitting a meeting transcript at [00:01:23] timestamps or a log file at ERROR: entries.






Sentence / Paragraph Chunking


📌 What it is

Instead of cutting text by a fixed size (tokens/characters), we split the text at natural linguistic boundaries — sentences or paragraphs.

This keeps each chunk as a complete thought, which helps retrieval quality and readability.

✅ Advantages

Keeps semantic meaning intact → Each chunk represents a full sentence or paragraph, so no “half sentences.”

More human-readable → Useful when chunks are shown back to the user (citations, context snippets).

Improves embedding quality → Embeddings capture complete ideas instead of partial fragments.

Reduces noise in retrieval → Since chunks are coherent, fewer irrelevant embeddings are returned.

Great for FAQs, transcripts, articles → Natural boundaries align with the way humans think and write.

📘 Example

Text:

"Artificial Intelligence is transforming industries worldwide. 
It is used in healthcare for diagnostics and in finance for fraud detection. 

Self-driving cars also rely on AI to navigate safely."

Sentence-based Chunking
Chunk 1: "Artificial Intelligence is transforming industries worldwide."
Chunk 2: "It is used in healthcare for diagnostics and in finance for fraud detection."
Chunk 3: "Self-driving cars also rely on AI to navigate safely."

Paragraph-based Chunking
Chunk 1: "Artificial Intelligence is transforming industries worldwide. 
It is used in healthcare for diagnostics and in finance for fraud detection."

Chunk 2: "Self-driving cars also rely on AI to navigate safely."


👉 Notice how each chunk is self-contained and meaningful. No sentence is split across chunks.

🚀 When to use

Knowledge bases, FAQs, articles, reports where answers are usually in sentence- or paragraph-sized units.

When chunks are shown directly to users (like “retrieved context”).

When semantic coherence matters more than token efficiency.

📌 Summary:

Sentence/Paragraph chunking keeps human meaning intact.

It may result in uneven chunk sizes (some short, some long), but embeddings are higher quality.


Documents are narrative or FAQ-like.

You want readable chunks (citations, UI display).

Content is relatively short (paragraphs under ~500 tokens).
⚠️ Risk: Some paragraphs may be too long for embeddings.

Example: Customer FAQ bot → each Q&A pair stored as a chunk.



Chunking Type	Main Disadvantage
Fixed-size	Cuts across sentences/meaning, context loss
Overlapping	More storage/compute, redundant retrieval
Sentence/Paragraph	Uneven sizes, long chunks may exceed token limits
Semantic	Expensive, complex, depends on embedding quality
Hierarchical	Needs structured docs, parsing complexity
Regex/Rule-based	Brittle, domain-specific, hard to generalize

