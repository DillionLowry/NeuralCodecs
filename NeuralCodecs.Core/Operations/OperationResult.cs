namespace NeuralCodecs.Core.Operations
{
    /// <summary>
    /// Represents the result of an operation, containing success status, result data, and error information.
    /// </summary>
    /// <typeparam name="T">The type of the result data.</typeparam>
    public record OperationResult<T>
    {
        /// <summary>
        /// Gets whether the operation was successful.
        /// </summary>
        public bool Success { get; init; }

        /// <summary>
        /// Gets the result data of the operation.
        /// </summary>
        public T Result { get; init; }

        /// <summary>
        /// Gets any error that occurred during the operation.
        /// </summary>
        public Exception Error { get; init; }

        /// <summary>
        /// Gets a message describing the operation outcome.
        /// </summary>
        public string Message { get; init; }

        /// <summary>
        /// Creates a successful operation result with the specified data.
        /// </summary>
        public static OperationResult<T> FromSuccess(T result) =>
            new() { Success = true, Result = result };

        /// <summary>
        /// Creates a failed operation result with the specified exception.
        /// </summary>
        public static OperationResult<T> FromError(Exception ex) =>
            new() { Success = false, Error = ex, Message = $"{ex.Message} {ex.InnerException?.Message}" };
    }
}