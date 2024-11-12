namespace NeuralCodecs.Torch.Utils;

public static class MathUtils
{
    public static long GCD(long a, long b)
    {
        while (b != 0)
        {
            long temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    public static long LCM(long a, long b)
    {
        return Math.Abs(a * b) / GCD(a, b);
    }

    public static double db_to_linear(double db)
    {
        return Math.Pow(10.0, db / 20.0);
    }

    public static double linear_to_db(double linear)
    {
        return 20.0 * Math.Log10(linear);
    }
}