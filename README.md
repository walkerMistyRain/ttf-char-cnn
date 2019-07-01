# ttf-char-cnn
pytorch cnn to training letters of ttf font
使用一個特定的TTF字體，產生訓練用的字元，來做一個自製的OCR。
希望可以幫助到有需要使用自己字體做訓練的人，請以此為參考，改成自己所需的模型。

環境，ubuntu18.04, my env for reference
pytorch                 >=  1.0.1           
opencv                    3.4.2   
scipy                     1.2.1   
matplotlib                3.1.0 
imageio                   2.5.0 
pillow                    6.0.0

pytorch安裝，可參考我的blogger文 : 
https://mistyraining.blogspot.com/2019/05/ubuntuanacondatensorflow-gpu-keras.html

訓練圖檔為36x36，字體大小從28~36，加上些許角度擺動及雜訊。   
字體：Harrington.ttf 網路下載   

cnnGpuTrain.py  訓練   
cnnGpuTest.py   測試訓練結果   
predOne.py      測試任一字元圖檔    

train.txt      訓練檔案列表與ID值  16688    
tets.txt       測試檔案列表與ID值  168    
train.tgz      訓練檔案目錄壓縮檔，訓練前須使用tar -zxvf train.tgz來解壓縮   
test.tgz       測試檔案目錄壓縮檔，測試前須使用tar -zxvf test.tgz來解壓縮    

genTrainLetters36x36.py   產生訓練檔與測試檔    

CNN模型與訓練結果   

GPU status =  True      
Net(  
  (conv1): Sequential(    
    (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))   
    (1): ReLU()    
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)   
  )    
  (conv2): Sequential(    
    (0): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))   
    (1): ReLU()    
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)    
  )     
  (fc1): Sequential(    
    (0): Linear(in_features=1296, out_features=200, bias=True)    
    (1): ReLU()     
  )    
  (fc2): Sequential(    
    (0): Linear(in_features=200, out_features=80, bias=True)   
    (1): ReLU()   
  )   
  (fc3): Linear(in_features=80, out_features=36, bias=True)   
)


epoch 1  
Train Loss: 0.017076, Acc: 0.395733     

epoch 2   
Train Loss: 0.004060, Acc: 0.842042    

epoch 3   
Train Loss: 0.001863, Acc: 0.925995   

epoch 4   
Train Loss: 0.001013, Acc: 0.956855   

epoch 5   
Train Loss: 0.000700, Acc: 0.969199   

epoch 6   
Train Loss: 0.000517, Acc: 0.978008   

epoch 7    
Train Loss: 0.000360, Acc: 0.983701    

epoch 8   
Train Loss: 0.000278, Acc: 0.987716    

epoch 9    
Train Loss: 0.000231, Acc: 0.989394    

epoch 10    
Train Loss: 0.000169, Acc: 0.993229    



測試結果    
Test Loss: 0.000823, Acc: 0.988095   



