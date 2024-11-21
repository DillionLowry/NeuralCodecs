using System.Text.RegularExpressions;

namespace NeuralCodecs.Core.Utils;

/// <summary>
/// Pattern matcher supporting wildcards
/// </summary>
public class WildcardPattern
{
    private readonly Regex _regex;

    public WildcardPattern(string pattern)
    {
        _regex = new Regex(
            "^" + Regex.Escape(pattern)
                      .Replace("\\*", ".*")
                      .Replace("\\?", ".") + "$",
            RegexOptions.IgnoreCase);
    }

    public bool IsMatch(string input) => _regex.IsMatch(input);
}