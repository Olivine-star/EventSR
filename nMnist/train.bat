@echo off
REM
call conda activate EventSRSNN

REM
cd /d D:\PycharmProjects\EventSR\SR-ES1\EventStream-SR-main\nMnist

REM
python trainNmnist.py ^
--bs 64 ^
--savepath "D:/PycharmProjects/EventSR-ckpt/ckpt/" ^
--epoch 30 ^
--showFreq 50 ^
--lr 0.1 ^
--cuda "0" ^
--j 4

pause
