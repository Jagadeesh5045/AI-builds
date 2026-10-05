# Fine-tuning

Fine-tuning continues training a pretrained model on task-specific data, shifting its
behaviour toward a narrower distribution. It is the right tool when the base model lacks a
skill or style, and the wrong tool when the problem is missing knowledge.

## Full fine-tuning vs PEFT

Full fine-tuning updates every weight, which is expensive and risks catastrophic forgetting
of general abilities. Parameter-efficient methods like LoRA freeze the base model and train
small low-rank adapter matrices instead, cutting trainable parameters by orders of magnitude.
LoRA adapters can be swapped at inference time, so one base model serves many tasks.

## Fine-tuning vs RAG

This is the most confused decision in applied AI. Fine-tuning teaches behaviour and style;
it is poor at injecting facts, because knowledge baked into weights goes stale and cannot
be cited. RAG supplies facts at query time with checkable sources. The practical rule: use
RAG for knowledge that changes or must be cited, fine-tuning for tone, format, and
domain-specific reasoning patterns. Many production systems use both.

## Data requirements

Fine-tuning lives or dies on data quality. A few hundred carefully curated examples beat ten
thousand noisy ones. Every training example should look like the real task, including edge
cases and the exact output format expected. If the eval set does not cover a behaviour, the
fine-tune will not learn it reliably.
