@echo off
REM
call conda activate EventSRSNN

REM
cd /d D:\PycharmProjects\pythonProject\EventSR\nMnist

REM
python trainNmnist.py ^
--bs 64 ^
--savepath "D:/PycharmProjects/EventSR-ckpt/ckpt2/" ^
--epoch 30 ^
--showFreq 50 ^
--lr 0.1 ^
--cuda "0" ^
--j 4

pause
