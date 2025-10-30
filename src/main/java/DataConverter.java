import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.Attribute;
import weka.core.DenseInstance;


import java.io.File;
import java.util.ArrayList;

public class DataConverter {

    public static void main(String[] args) {
        // Dönüştürülecek CSV dosyasının yolu.
        String csvFilePath = "C:/Users/seyma/IdeaProjects/Hisse_Senedi_Tahmini/src/main/resources/Aselsan_Stock Price_History.csv";

        // Oluşturulacak ARFF dosyasının kaydedileceği yol.
        String arffFilePath = "aselsan_data.arff";

        try {
            // CSV dosyasını okumak için CSVLoader kullanıyoruz.
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(csvFilePath));
            loader.setFieldSeparator(","); // CSV ayırıcısı
            loader.setNoHeaderRowPresent(true); // Başlık satırı YOK



            Instances data = loader.getDataSet();

            // Yeni öznitelik isimleri ile yeni bir Instances nesnesi oluştur
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("Price"));
            attributes.add(new Attribute("Open"));
            attributes.add(new Attribute("High"));
            attributes.add(new Attribute("Low"));
            attributes.add(new Attribute("Vol"));
            attributes.add(new Attribute("Change"));

            Instances newData = new Instances("Aselsan_Stock_Data", attributes, data.numInstances());

            // Verileri kopyala
            for (int i = 0; i < data.numInstances(); i++) {
                double[] values = new double[data.numAttributes()];
                for (int j = 0; j < data.numAttributes(); j++) {
                    values[j] = data.instance(i).value(j);
                }
                newData.add(new DenseInstance(1.0, values));
            }



            // Sınıf indeksini manuel olarak ayarla (6. sütun - Change)
            newData.setClassIndex(5);


            // Veriyi ARFF formatında kaydetmek için ArffSaver kullanıyoruz.
            ArffSaver saver = new ArffSaver();
            saver.setInstances(newData);
            saver.setFile(new File(arffFilePath));
            saver.writeBatch();


            // Veriyi belirtilen yola kaydediyoruz.
            saver.setFile(new File(arffFilePath));
            saver.writeBatch();

            System.out.println("Veri başarıyla ARFF formatına dönüştürüldü!");
            System.out.println("Dosya: " + arffFilePath);

        } catch (Exception e) {
            System.out.println("Bir hata oluştu: " + e.getMessage());
            e.printStackTrace();
        }
    }
}