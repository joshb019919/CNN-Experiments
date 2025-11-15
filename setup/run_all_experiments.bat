@echo off
echo ========================================
echo CSC 737 Assignment 3 - Experiment Runner
echo ========================================

REM Create output directory
mkdir runs

echo.
echo ---- DEPTH COMPARISON ----
echo Running BaseNet-10 (SGD)...
python train_cnn.py --model base --variant 10 --optimizer sgd --save_dir runs

echo Running BaseNet-16 (SGD)...
python train_cnn.py --model base --variant 16 --optimizer sgd --save_dir runs


echo.
echo ---- ARCHITECTURE COMPARISON ----
echo Running BaseNet-16 (SGD)...
python train_cnn.py --model base --variant 16 --optimizer sgd --save_dir runs

echo Running ResNet-18 (SGD)...
python train_cnn.py --model resnet --optimizer sgd --save_dir runs


echo.
echo ---- OPTIMIZER COMPARISON ----
echo Running ResNet-18 (SGD)...
python train_cnn.py --model resnet --optimizer sgd --save_dir runs

echo Running ResNet-18 (Adam)...
python train_cnn.py --model resnet --optimizer adam --save_dir runs

echo.
echo All experiments completed!
pause
