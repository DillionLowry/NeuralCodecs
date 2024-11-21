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

    public static double HertzToMel(double hz)
    {
        return 2595.0 * Math.Log10(1.0 + hz / 700.0);
    }

    public static double MelToHertz(double mel)
    {
        return 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);
    }
}