##IMPORTANT : Run the code using the Bash command not the sh command .So to run the code do the following: bash runmanycode.sh
##This is a shell script to run multiple instance of K and n 
##Step 1: Create the necessary folder as required by the code 
##Step 1 : Create the folder where the output is going to be : 
##Step2 : Remove the replication folder if already exists 
##Step 3 : Run the python Script


for ((K=305;K<=305;K=K+5))
do
if [ $K -eq 0 ]
   then $K = $K+1
fi
##Step1
  mkdir "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5"
  mkdir "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5/weight_transfer"
 touch "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5/weight_transfer/opts.txt"


echo "
-d mnist
-dp /home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/datasets/mnist 
--t1_train 8000
--t1_valid 3000
-k $K
-n 5
--t2_test 10000
-e 500
-bs 2048
-lr 0.005
-p 20
-esr 0.01
-r 1234
--replications 1
-g 0
-ctl /cpu:0
-sd /home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5/weight_transfer
-lf /home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5/weight_transfer/log.txt
weight_transfer
" >> "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5/weight_transfer/opts.txt"


  mkdir "/home/abhishek/Desktop/V2_MNIST_$K""_5"
  rm -rf "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5/weight_transfer/replication1"
python3 main.py @"trained_models/mnist/mnist_$K""_5/weight_transfer/opts.txt"

done

#mkdir /home/abhishek/Desktop/V2_MNIST_1_5 




#rm -rf /home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/#trained_models/#mnist/mnist_1_5/weight_transfer/replication1


#python3 main.py @trained_models/mnist/mnist_1_5/weight_transfer/opts.txt


#mkdir /home/abhishek/Desktop/V2_MNIST_2_5 
#rm -rf /home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/#trained_models/mnist/mnist_2_5/weight_transfer/replication1
#python3 main.py @trained_models/mnist/mnist_2_5/weight_transfer/opts.txt




