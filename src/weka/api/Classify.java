package weka.api;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

public class Classify {

	static String filePath = "input.json";

	public static void main(String[] args) throws Exception {
		
		System.out.println("Classify");
		
		try {
			// BufferedReader ile dosyayý okuyoruz
			BufferedReader reader = new BufferedReader(new FileReader(filePath));

			// JSON parser ile okudumuðumuz dosyayý parse edip bir jsonObj oluþturuyoruz
			JSONParser jsonParser = new JSONParser();
			JSONObject jsonObject = (JSONObject) jsonParser.parse(reader);

			// String question = (String) jsonObject.get("question");
			// System.out.println(jsonObject);

			FileWriter arff = new FileWriter("new.arff");

			arff.append("@relation set2-emreaydin\n");
			arff.append("@attribute question string\n");
			arff.append(
					"@attribute class { rsp1, rsp2, rsp3, rsp4, rsp5, rsp6, rsp7, rsp8, rsp9, rsp10, rsp11, rsp12, rsp13 ,rsp14, rsp15, rsp16, rsp17, rsp18, rsp19, rsp20, rsp21, rsp22, rsp23, rsp24, rsp25, rsp26, rsp27, rsp28 ,rsp29, rsp30, rsp31, rsp32, rsp33, rsp34, rsp35, rsp36, rsp37, rsp38, rsp39, rsp40, rsp41, rsp42, rsp43, rsp44, rsp45, rsp46, rsp47, rsp48, rsp49, rsp50, rsp51, rsp52, rsp53, rsp54, rsp55, rsp56, rsp57, rsp58, rsp59, rsp60, rsp61, rsp62, rsp63, rsp64, rsp65, rsp66, rsp67, rsp68, rsp69, rsp70, rsp71, rsp72, rsp73, rsp74, rsp75, rsp76, rsp77, rsp78, rsp79, rsp80, rsp81, rsp82, rsp83, rsp84, rsp85, rsp86, rsp87, rsp88, rsp89, rsp90, rsp91, rsp92, rsp93, rsp94, rsp95, rsp96, rsp97, rsp98, rsp99, rsp100 ,rsp101, rsp103, rsp104, rsp105, rsp106, rsp107, rsp108, rsp109, rsp110, rsp111, rsp112, rsp113, rsp115, rsp116, rsp117, rsp118}\n");
			arff.append("@data\n");

			String question = (String) jsonObject.get("question");

			arff.append("'" + question + "'" + ",");
			arff.append("?");

			arff.flush();
			arff.close();

		} catch (FileNotFoundException e) {

		}

		NaiveBayes NBM = (NaiveBayes) weka.core.SerializationHelper.read("tuchatbot.model");

		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader("dataset.arff"));
		Instances train = new Instances(breader);
		train.setClassIndex(train.numAttributes() - 1);

		breader = new BufferedReader(new FileReader("new.arff"));
		Instances test = new Instances(breader);
		test.setClassIndex(test.numAttributes() - 1);

		Instance newInstance = test.instance(0);
		double PredictVal = NBM.classifyInstance(newInstance);
		System.out.println(PredictVal);

		Instances labeled = new Instances(test);

		// label instances
		for (int i = 0; i < test.numInstances(); i++) {
			double clsLabel = NBM.classifyInstance(test.instance(i));
			labeled.instance(i).setClassValue(clsLabel);
		}

		System.out.println(labeled.instance(0));

		breader.close();

	}

}

/*
 * //NaiveBayes NBM = new NaiveBayes(); //NBM =
 * (NaiveBayes)weka.core.SerializationHelper.read("tuchatbot.model");
 * 
 * NaiveBayes NBM =
 * (NaiveBayes)weka.core.SerializationHelper.read("tuchatbot.model");
 * 
 * 
 * BufferedReader breader = null; breader = new BufferedReader(new
 * FileReader("set2-emreaydin.arff")); Instances train = new Instances(breader);
 * train.setClassIndex(train.numAttributes()-1);
 * 
 * breader = new BufferedReader(new FileReader("new.arff")); Instances test =
 * new Instances(breader); test.setClassIndex(test.numAttributes()-1);
 * 
 * Instance newInstance=test.instance(0); double
 * PredictVal=NBM.classifyInstance(newInstance); System.out.println(PredictVal);
 */

/*
 * BufferedReader breader = null; breader = new BufferedReader(new
 * FileReader("set2-emreaydin.arff")); Instances train = new Instances(breader);
 * train.setClassIndex(train.numAttributes()-1);
 * 
 * breader = new BufferedReader(new FileReader("new.arff")); Instances test =
 * new Instances(breader); test.setClassIndex(test.numAttributes()-1);
 * 
 * NaiveBayesMultinomialText tree = new NaiveBayesMultinomialText();
 * 
 * FilteredClassifier nesne = new FilteredClassifier(); StringToWordVector
 * filter = new StringToWordVector(); nesne.setFilter(filter);
 * nesne.setClassifier(new J48()); nesne.buildClassifier(train);
 * 
 * tree.buildClassifier(train); Instances labeled = new Instances(test);
 * 
 * //label instances for(int i = 0; i < test.numInstances(); i++) { double
 * clsLabel = nesne.classifyInstance(test.instance(i));
 * labeled.instance(i).setClassValue(clsLabel); }
 * 
 * System.out.println(labeled.instance(0));
 * 
 * breader.close();
 */
