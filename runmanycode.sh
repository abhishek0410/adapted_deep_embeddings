##IMPORTANT : Run the code using the Bash command not the sh command .So to run the code do the following: bash runmanycode.sh
##This is a shell script to run multiple instance of K and n 

##Step 1: Create the necessary folder as required by the code (and delete if alrady exists )
##Step 2 : Create the folder where the output is going to be : 
##Step 3 : Remove the replication folder if already exists 
##Step 4 : Run the python Script


for ((K=350;K<=1000;K=K+50))
do
if [ $K -eq 0 ]
   then $K = $K+1
fi
##Step1
##Deleting the file if it exists already : 
DIR=/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5
if [ -d "$DIR" ]; then
    echo "Removing Lock"
    rm -rf "$DIR"
fi

##Create the necessary folder 	
  mkdir "/home/abhishek/Desktop/Results_Exp1"
  mkdir "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/MODEL_exp1"
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


##Step 2 : 
  mkdir "/home/abhishek/Desktop/Results_Exp1"
##Step 3 : 
  rm -rf "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_$K""_5/weight_transfer/replication1"

##Step 4 :
python3 main.py @"trained_models/mnist/mnist_$K""_5/weight_transfer/opts.txt"

done






