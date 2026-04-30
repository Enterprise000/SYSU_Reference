public class TaxRange {
    private final int low;
    private final int high;
    private double rate;

    public TaxRange(int low, int high, double rate){
        this.high = high;
        this.low = low;
        this.rate = rate;
    }

    public int getLow(){
        return low;
    }

    public int getHigh(){
        return high;
    }

    public double getRate(){
        return rate;
    }

}
