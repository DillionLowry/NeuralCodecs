using System;
using System.Text.RegularExpressions;

namespace NeuralCodecs.Core.Utils
{
    /// <summary>
    /// Provides functionality for matching strings against wildcard patterns.
    /// </summary>
    public class WildcardPattern
    {
        private readonly Regex _regex;

        /// <summary>
        /// Initializes a new instance of the WildcardPattern class.
        /// </summary>
        /// <param name="pattern">The wildcard pattern to match against.</param>
        public WildcardPattern(string pattern)
        {
            pattern = pattern ?? throw new ArgumentNullException(nameof(pattern));
            
            // Convert wildcard pattern to regex
            string regexPattern = "^" + Regex.Escape(pattern)
                .Replace("\\*", ".*")
                .Replace("\\?", ".") + "$";
                
            _regex = new Regex(regexPattern, RegexOptions.IgnoreCase);
        }

        /// <summary>
        /// Determines if the input string matches the wildcard pattern.
        /// </summary>
        /// <param name="input">The string to test.</param>
        /// <returns>True if the input matches the pattern, false otherwise.</returns>
        public bool IsMatch(string input)
        {
            if (input == null)
                return false;
                
            return _regex.IsMatch(input);
        }
    }
}