//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
import java.util.*;
public class Main {
    public static void main(String[] args) {

        System.out.print("若您要使用默认计算模式，请输入0；如果您要自定义个税起征点和各级税率，请输入1：");
        Scanner scanner = new Scanner(System.in);
        int mode = scanner.nextInt();
        List<TaxRange> TexRanges = new ArrayList<>();
        if(mode !=0 && mode !=1){
            System.out.print("非法输入！");
            return;
        }
        if(mode == 0){
            TaxCmpt calculator = new TaxCmpt(5000);
            calculator.addRange(new TaxRange(0, 3000, 0.03));
            calculator.addRange(new TaxRange(3000, 12000, 0.10));
            calculator.addRange(new TaxRange(12000, 25000, 0.20));
            calculator.addRange(new TaxRange(25000, 35000, 0.25));
            calculator.addRange(new TaxRange(35000, 55000, 0.30));
            calculator.addRange(new TaxRange(55000, 80000, 0.35));
            calculator.addRange(new TaxRange(80000, Integer.MAX_VALUE, 0.45));
            System.out.print("请输入您的收入: ");
            double income = scanner.nextDouble();
            if(income <0){
                System.out.print("收入不能是负数！");
                return;
            }
            double tax = calculator.compute(income);
            System.out.println("应缴纳的个人所得税为: " + tax);
            return;
        }
        else{
            System.out.println("请输入个税起征点: ");
            double standpoint = scanner.nextDouble();
            if(standpoint < 0){
                System.out.print("起征点不能是负数！");
                return;
            }
            TaxCmpt calculator = new TaxCmpt(standpoint);
            double[] RateArray = new double[7];
            for(int i=1; i<8;i++){
                System.out.println("请输入第" + i + "个征税级别的税率: ");
                double rate= scanner.nextDouble();
                if(rate < 0){
                    System.out.print("税率不能是负数！");
                    return;
                }
                RateArray[i-1] = rate;
            }
            calculator.addRange(new TaxRange(0, 3000, RateArray[0]));
            calculator.addRange(new TaxRange(3000, 12000, RateArray[1]));
            calculator.addRange(new TaxRange(12000, 25000, RateArray[2]));
            calculator.addRange(new TaxRange(25000, 35000, RateArray[3]));
            calculator.addRange(new TaxRange(35000, 55000, RateArray[4]));
            calculator.addRange(new TaxRange(55000, 80000, RateArray[5]));
            calculator.addRange(new TaxRange(80000, Integer.MAX_VALUE, RateArray[6]));
            System.out.print("请输入您的收入: ");
            double income = scanner.nextDouble();
            if(income <0){
                System.out.print("收入不能是负数！");
                return;
            }
            double tax = calculator.compute(income);
            System.out.println("应缴纳的个人所得税为: " + tax);
            return;
        }
    }
}
