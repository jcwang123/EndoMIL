# EndoMIL
**Foundation Models Assesses Gastric Cancer Risk from Endoscopic Findings**

EndoMIL aims to assess GC risk via endoscopic results of a patient according to the Kyoto classification. 

The workfolow of EndoMIL is illustrated as follows:
![image](https://github.com/jcwang123/EndoMIL/blob/main/workflow.png)


We apply Multi-instance Learning to define endoscopic results, and leverage the power of multiple foundation models to EndoMIL.

The architecture of EndoMIL is illustrated as follows:
![image](https://github.com/jcwang123/EndoMIL/blob/main/model.png)

Results demonstrate the outsanding performance of EndoMIL, verify its ability in clinical settings. 
Compared to exsiting methods since it does best in evaluating GC risk, it achieve an accuracy of ***78.45%*** and a scoring error of ***1.0431***.

In addition, you could download data and checkpoints from https://pan.baidu.com/s/1NX5_Cd63MnlfAhnv2pBp0g?pwd=6666, password=6666.
The pre-trained Endo-FM could be found from https://github.com/med-air/Endo-FM?tab=readme-ov-file.

