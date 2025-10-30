import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Evaluation;


import java.io.File;
import java.util.Random;
import java.util.Scanner;
import java.util.ArrayList;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;



    public class StockPredictor {

        public static void main(String[] args) {
            try {
                // Adım 1: ARFF veri dosyasını yükle
                ArffLoader loader = new ArffLoader();
                loader.setSource(new File("aselsan_data.arff"));
                Instances data = loader.getDataSet();

                // Adım 2: Veri setini eğitime hazırla
                // Veri setindeki son sütun (Close fiyatı), tahmin etmek istediğimiz değerdir.
                // "Price" sütununu hedef olarak belirliyoruz
                int priceIndex = data.attribute("Price").index();
                data.setClassIndex(priceIndex);



                // Adım 3: Lineer Regresyon modelini oluştur ve eğit
                LinearRegression model = new LinearRegression();
                model.buildClassifier(data);

                // Adım 4: Modelin performansını değerlendir
                // Modelin ne kadar iyi çalıştığını görmek için bir değerlendirme yapıyoruz.
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(model, data, 10, new Random(1)); // 10 katmanlı çapraz doğrulama

                // Adım 5: Sonuçları konsola yazdır
                System.out.println(eval.toSummaryString("\nSonuçlar\n", false));

                // Eğer istersen, eğitilen modeli konsola yazdırabilirsin
                System.out.println("\nEğitilen Model:\n" + model);

                // Adım 6: Yeni verilerle tahmin yap
                predictStockPrice(model,data);


            } catch (Exception e) {
                System.err.println("Bir hata oluştu: " + e.getMessage());
            }

        }


        private static void predictStockPrice(LinearRegression model, Instances data) {
            try {
                Scanner scanner = new Scanner(System.in);

                System.out.println("\n--- Tekil Tahmin Modu ---");
                System.out.println("Lütfen aşağıdaki değerleri girin:");

                System.out.print("Açılış (Open) fiyatı: ");
                double open = scanner.nextDouble();

                System.out.print("En Yüksek (High) fiyat: ");
                double high = scanner.nextDouble();

                System.out.print("En Düşük (Low) fiyat: ");
                double low = scanner.nextDouble();

                System.out.print("Hacim (Vol): ");
                double vol = scanner.nextDouble();

                // Yeni tahmin için bir Instance (veri örneği) oluştur
                // Bu örnek, modelin eğitildiği veri setiyle aynı formatta olmalı
                // Modelin eğitildiği veri setinin yapısını kullanıyoruz
                // Tahmin için bir veri örneği oluştur
                Instances predictionData = new Instances(data, 0);

                // Yeni tahmin için bir Instance (veri örneği) oluştur
                // Burada modelin eğitildiği tüm nitelikleri kullanıyoruz.
                DenseInstance newInstance = new DenseInstance(predictionData.numAttributes());

                // Değerleri doğru niteliklere atıyoruz
                newInstance.setValue(predictionData.attribute("Open"), open);
                newInstance.setValue(predictionData.attribute("High"), high);
                newInstance.setValue(predictionData.attribute("Low"), low);
                newInstance.setValue(predictionData.attribute("Vol"), vol);

                // Modelimizin ihtiyaç duyduğu ancak kullanıcının girmeyeceği diğer nitelikler için
                // bu değerleri atlıyoruz veya varsayılan bir değer veriyoruz.
                // Örneğin, "Change" niteliği varsa, bu niteliğe 0 atayabiliriz.
                if (predictionData.attribute("Change") != null) {
                    newInstance.setValue(predictionData.attribute("Change"), 0);
                }

                // Tahmin edilecek olan "Price" niteliğini boş bırakıyoruz
                newInstance.setMissing(predictionData.attribute("Price"));

                // Yeni örneği tahmin verisi setine ekliyoruz ve bu seti kullanıyoruz
                predictionData.add(newInstance);
                newInstance.setDataset(predictionData);


                // Tahmin yap
                double prediction = model.classifyInstance(newInstance);

                System.out.println("\nTahmin edilen kapanış fiyatı: " + prediction);


                //Tahmin sonucunu dosyaya kaydet
                savePredictionToFile(open, high, low, vol, prediction);

            } catch (Exception e) {
                System.err.println("Tahmin yapılırken bir hata oluştu: " + e.getMessage());
            }
        }


        private static void savePredictionToFile(double open, double high, double low, double vol, double prediction) {
            try {
                // "predictions.txt" adında bir dosya oluştur veya varsa sonuna ekle
                BufferedWriter writer = new BufferedWriter(new FileWriter("predictions.txt", true));

                // Tahmin verilerini, kolay okunacak bir formatta dosyaya yaz
                writer.write(String.format("Girdiler -> Open: %.2f, High: %.2f, Low: %.2f, Vol: %.2f | Tahmin Edilen Fiyat: %.2f\n",
                        open, high, low, vol, prediction));

                writer.close();
                System.out.println("Tahmin sonucu 'predictions.txt' dosyasına kaydedildi.");

            } catch (IOException e) {
                System.err.println("Dosya yazılırken bir hata oluştu: " + e.getMessage());
            }
        }
    }




