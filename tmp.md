Hi Chen and Mrinal,

Due to the huge gap between GPU and IPU on throughput in MLPerf Bert SQuAD inference, The one possible solution is the effective transformer. However, I get some questions.

1. is it necessary that I implement the packed BERT inference code by myself or just waiting for that from the other team?

2. if we wait for the other team implementation, can I know the working process and deadline because I saw the [T31547](https://phabricator.sourcevertex.net/T31547) is created and still open.

3. Can I have the example code for TF2 or TF1 packed Bert inference code since I saw [T31544](https://phabricator.sourcevertex.net/T31544) is created.

4. if need me to implement packed BERT inference code on TF2 with `Effective Transformer Keras Layer` ([T35014](https://phabricator.sourcevertex.net/T35014)), How can I get the TF2 wheel file? I understand Jack can not build the wheel file for me and I try to pull the code and build it for myself, but how can I get the permission (I fail to run `gc-view pull`, it shows I do not have the permission)? or Can I get TF2 wheel file include `Effective Transformer Keras Layer` from the regular SDK builds? [image]

5. if need me to implement `Effective Transformer` code on PopART with `packedDataBlockOp` ([T35240](https://phabricator.sourcevertex.net/T35240)), which version SDK should I use?

Regards

Scott