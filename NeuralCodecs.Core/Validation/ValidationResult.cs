namespace NeuralCodecs.Core.Validation
{
    /// <summary>
    /// Represents the result of a validation operation, containing validation status and any error messages.
    /// </summary>
    public record ValidationResult
    {
        /// <summary>
        /// Gets whether the validation was successful (no errors present).
        /// </summary>
        public bool IsValid => Errors.Count == 0;

        /// <summary>
        /// Gets the list of validation error messages.
        /// </summary>
        public List<string> Errors { get; init; } = new();

        /// <summary>
        /// Creates a successful validation result with no errors.
        /// </summary>
        public static ValidationResult Success() => new();

        /// <summary>
        /// Creates a failed validation result with the specified error messages.
        /// </summary>
        /// <param name="errors">The error messages to include.</param>
        public static ValidationResult Failed(params string[] errors) =>
            new() { Errors = errors.ToList() };
    }
}