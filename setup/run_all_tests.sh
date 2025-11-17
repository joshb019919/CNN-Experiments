#!/bin/bash

echo
echo "---- RUNNING TESTS ----"
echo "Running BaseNet-10 test..."
python3 tests/smoke_test.py --model base --variant 10

echo
echo "Running ResNet test..."
python3 tests/smoke_test.py --model resnet

echo
echo "All tests complete!"
