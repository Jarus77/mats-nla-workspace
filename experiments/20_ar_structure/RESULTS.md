# Exp 20: what the NLA reconstructor keeps (ordered span, bag of the same tokens, the readers' own words)

Neel (27 Oct): "understanding why o-lens and NLAs fail to capture that thing would be interesting; I recommend studying
the AR and seeing if it captures any of that structure." Exps 14 to 16 did this on two position rows; this runs the same
question on the eight natural rows of exp 19 (selection split, 160 items per row, layer-42 activations already captured)
with four kinds of input to the reconstructor, scored by exp 14's frozen-probe AUROC against the true activations:

- the true last w tokens, w = 4, 8, 16, 32, 64, 128 (how much structure the window carries, and whether more text helps)
- the same 32 tokens in random order (vocabulary kept, structure broken)
- Oracle Lens's own free readout of the vector (k = 1, layer 42)
- the NLA's own explanation of the vector (k = 1)
- the model's own layer-42 activation when it reads the last w tokens alone, no reconstructor (the image of the window map: what any text-emitting reader could at most re-induce)

If the ordered span keeps the state and the shuffled bag and the readouts do not, the readers' failure is located in what
the reconstruction objective rewards: content, not order. App `v7qd` (`modal20.py`); one H100 pass for the Oracle Lens
readouts, one H200 pass for the NLA and the reconstructions.

## Results

(pending)
