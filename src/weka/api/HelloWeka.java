package weka.api;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class HelloWeka {

	public static void main(String [] args) throws Exception {
	
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("deneme.arff"));
		
		Instances train = new Instances(breader);
		train.setClassIndex(train.numAttributes()-1);
		
		breader.close();
		
		NaiveBayes nB = new NaiveBayes();
		nB.buildClassifier(train);
		
		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(nB, train, 10, new Random(1));
		
		System.out.println(eval.toSummaryString("\n Results \n--------------\n",true));
		System.out.println(eval.fMeasure(1)+" "+eval.precision(1)+ " "+ eval.recall(1));
		
		
		
	}
}
