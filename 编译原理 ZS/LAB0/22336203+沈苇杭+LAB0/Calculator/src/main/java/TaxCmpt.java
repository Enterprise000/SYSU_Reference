import java.util.*;
public class TaxCmpt {
    private List<TaxRange> TaxRanges;
    private double StartPoint;

    public TaxCmpt(double StartPoint){
        TaxRanges = new ArrayList<>();
        this.StartPoint = StartPoint;
    }

    public void addRange(TaxRange taxrange){
        TaxRanges.add(taxrange);
    }

    public double compute(double money){
        double pay = money - StartPoint;
        double tax = 0;
        for(TaxRange taxrange : TaxRanges){
            if(pay > taxrange.getLow()){
                if(pay < taxrange.getHigh()){
                    tax = tax + (pay - taxrange.getLow())* taxrange.getRate();
                }
                else{
                    tax = tax + (taxrange.getHigh() - taxrange.getLow()) * taxrange.getRate();
                }
            }
        }
        return tax;
    }
}
