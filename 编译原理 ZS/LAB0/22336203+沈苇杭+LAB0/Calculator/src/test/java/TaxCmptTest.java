import org.junit.jupiter.api.Test; // JUnit 5
import static org.junit.jupiter.api.Assertions.assertEquals;
public class TaxCmptTest {
   @Test
    public void TestTax(){
       TaxCmpt texCmpt = new TaxCmpt(5000);
       texCmpt.addRange(new TaxRange(0, 3000, 0.03));
       texCmpt.addRange(new TaxRange(3000, 12000, 0.10));
       texCmpt.addRange(new TaxRange(12000, 25000, 0.20));
       texCmpt.addRange(new TaxRange(25000, 35000, 0.25));
       texCmpt.addRange(new TaxRange(35000, 55000, 0.30));
       texCmpt.addRange(new TaxRange(55000, 80000, 0.35));
       texCmpt.addRange(new TaxRange(80000, Integer.MAX_VALUE, 0.45));
       double result1 = texCmpt.compute(6000);
       assertEquals(30, result1, "采用默认计算模式，起征点为5000，收入为6000，应该交税30");
       double result2 = texCmpt.compute(9000);
       assertEquals(190, result2, "采用默认计算模式，起征点为5000，收入为9000，应该交税190");
      double result3 = texCmpt.compute(18000);
      assertEquals(1190, result3, "采用默认计算模式，起征点为5000，收入为18000，应该交税1190");
      double result4 = texCmpt.compute(35000);
      assertEquals(4840, result4, "采用默认计算模式，起征点为5000，收入为35000，应该交税4840");
      double result5 = texCmpt.compute(50000);
      assertEquals(9090, result5, "采用默认计算模式，起征点为5000，收入为50000，应该交税9090");
      double result6 = texCmpt.compute(70000);
      assertEquals(15590, result6, "采用默认计算模式，起征点为5000，收入70000，应该交税15590");
      double result7 = texCmpt.compute(100000);
      assertEquals(27590, result7, "采用默认计算模式，起征点为5000，收入100000，应该交税27590");
   }
   @Test
   public void TestLowerThanStartPoint(){
       TaxCmpt texCmpt = new TaxCmpt(5000);
      texCmpt.addRange(new TaxRange(0, 3000, 0.03));
      texCmpt.addRange(new TaxRange(3000, 12000, 0.10));
      texCmpt.addRange(new TaxRange(12000, 25000, 0.20));
      texCmpt.addRange(new TaxRange(25000, 35000, 0.25));
      texCmpt.addRange(new TaxRange(35000, 55000, 0.30));
      texCmpt.addRange(new TaxRange(55000, 80000, 0.35));
      texCmpt.addRange(new TaxRange(80000, Integer.MAX_VALUE, 0.45));
      double result = texCmpt.compute(2000);
      assertEquals(0, result, "采用默认计算模式，起征点为5000，收入为2000，应该交税0");
   }
   @Test
   public void TestEqualToStartPoint(){
       TaxCmpt texCmpt = new TaxCmpt(5000);
      texCmpt.addRange(new TaxRange(0, 3000, 0.03));
      texCmpt.addRange(new TaxRange(3000, 12000, 0.10));
      texCmpt.addRange(new TaxRange(12000, 25000, 0.20));
      texCmpt.addRange(new TaxRange(25000, 35000, 0.25));
      texCmpt.addRange(new TaxRange(35000, 55000, 0.30));
      texCmpt.addRange(new TaxRange(55000, 80000, 0.35));
      texCmpt.addRange(new TaxRange(80000, Integer.MAX_VALUE, 0.45));
      double result = texCmpt.compute(5000);
      assertEquals(0, result, "采用默认计算模式，起征点为5000，收入为5000，应该交税0");
   }
}
