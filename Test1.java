package com.example.meenu.application_detection;

        import android.os.Bundle;

        import com.example.meenu.application_detection.ModelClassifier;
        import com.example.meenu.application_detection.ModelGenerator;
//import weka.classifiers.functions.MultilayerPerceptron;
        import weka.classifiers.functions.LibSVM;
        import weka.core.Debug;
        import weka.core.Instances;
        import weka.filters.Filter;
        import weka.filters.unsupervised.attribute.Normalize;
        import android.support.v7.app.AppCompatActivity;
        import android.os.Bundle;
        import android.util.Log;
        import android.view.View;
        import android.widget.Button;
        import android.widget.TextView;


public class Test1  extends AppCompatActivity{
    TextView tv_text;
    public static final String DATASETPATH = "E:/Internship_Project/Android_App/ap/src/main/java/com/example/meenu/application_detection/All.csv";
    public static final String MODElPATH = "/Users/Emaraic/Temp/ml/model.bin";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test1);
        if (savedInstanceState != null) {
            Log.d("STATE", savedInstanceState.toString());
        }
    }


    //public static final String DATASETPATH = "/Users/Emaraic/Temp/ml/iris.2D.arff";


    void main1() throws Exception {

        ModelGenerator mg = new ModelGenerator();

        Instances dataset = mg.loadDataset(DATASETPATH);

        Filter filter = new Normalize();

        // divide dataset to train dataset 80% and test dataset 20%
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;

        dataset.randomize(new Debug.Random(1));// if you comment this line the accuracy of the model will be droped from 96.6% to 80%

        //Normalize dataset
        filter.setInputFormat(dataset);
        Instances datasetnor = Filter.useFilter(dataset, filter);

        Instances traindataset = new Instances(datasetnor, 0, trainSize);
        Instances testdataset = new Instances(datasetnor, trainSize, testSize);

        // build classifier with train dataset
        LibSVM ann =  (LibSVM) mg.buildClassifier(traindataset);

        // Evaluate classifier with test dataset
        String evalsummary = mg.evaluateModel(ann, traindataset, testdataset);
        Log.d("Evaluation: " ,evalsummary);

        //Save model
        // mg.saveModel(ann, MODElPATH);

        //classifiy a single instance
        //ModelClassifier cls = new ModelClassifier();
        //String classname =cls.classifiy(Filter.useFilter(cls.createInstance(1.6, 0.2, 0), filter), MODElPATH);
        //System.out.println("\n The class name for the instance with petallength = 1.6 and petalwidth =0.2 is  " +classname);


    }

}


