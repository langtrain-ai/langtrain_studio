"""LangTune: LoRA fine-tuning that is practical on a CPU.

The speed comes from training LoRA adapters on only the top K transformer
layers of a frozen base model. The hidden states entering layer L-K never
change during training, so they are computed once and cached; every later
epoch runs (and backpropagates through) only the top K layers.
"""

__version__ = "0.1.0"
