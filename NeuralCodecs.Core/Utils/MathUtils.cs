namespace NeuralCodecs.Torch.Utils;

public static class MathUtils
{
    /// <summary>
    /// Calculates Greatest Common Divisor (GCD) of two numbers.
    /// </summary>
    /// <param name="a">The first number</param>
    /// <param name="b">The second number</param>
    /// <returns>The greatest common divisor of the two numbers</returns>
    public static long GCD(long a, long b)
    {
        if (a == 0 && b == 0) throw new ArgumentException("Both numbers cannot be 0");
        a = Math.Abs(a);
        b = Math.Abs(b);
        while (b != 0)
        {
            long temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    /// <summary>
    /// Calculates Least Common Multiple (LCM) of two numbers.
    /// </summary>
    /// <param name="a">The first number</param>
    /// <param name="b">The second number</param>
    /// <returns>The least common multiple of the two numbers</returns>
    public static long LCM(long a, long b)
    {
        return Math.Abs(a * b) / GCD(a, b);
    }

    /// <summary>
    /// Converts decibel value to linear scale.
    /// </summary>
    /// <param name="db">The decibel value</param>
    /// <returns>The linear scale value</returns>
    public static double DecibelToLinear(double db)
    {
        return Math.Pow(10.0, db / 20.0);
    }

    /// <summary>
    /// Converts linear scale value to decibel.
    /// </summary>
    /// <param name="linear">The linear scale value</param>
    /// <returns>The decibel value</returns>
    public static double LinearToDecibel(double linear)
    {
        if (linear <= 0) throw new ArgumentException("Linear value must be positive");
        return 20.0 * Math.Log10(linear);
    }

    /// <summary>
    /// Converts frequency in Hertz to Mel scale.
    /// </summary>
    /// <param name="hz">The frequency in Hertz</param>
    /// <returns>The frequency in Mel scale</returns>
    public static double HertzToMel(double hz)
    {
        return 2595.0 * Math.Log10(1.0 + (hz / 700.0));
    }

    /// <summary>
    /// Converts frequency in Mel scale to Hertz.
    /// </summary>
    /// <param name="mel">The frequency in Mel scale</param>
    /// <returns>The frequency in Hertz</returns>
    public static double MelToHertz(double mel)
    {
        return 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);
    }

    public static double Erf(double x)
    {
        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;

        // Save the sign of x
        int sign = 1;
        if (x < 0)
            sign = -1;
        x = Math.Abs(x);

        // A&S formula 7.1.26
        double t = 1.0 / (1.0 + (p * x));
        double y = 1.0 - (((((((((a5 * t) + a4) * t) + a3) * t) + a2) * t) + a1) * t * Math.Exp(-x * x));

        return sign * y;
    }
}