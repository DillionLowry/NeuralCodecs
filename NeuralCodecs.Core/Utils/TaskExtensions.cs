namespace NeuralCodecs.Core.Utils
{
    /// <summary>
    /// Provides extension methods for Task-based operations.
    /// </summary>
    internal static class TaskExtensions
    {
        /// <summary>
        /// Applies a timeout to the execution of a ValueTask.
        /// </summary>
        /// <typeparam name="T">The type of the task result.</typeparam>
        /// <param name="task">The task to apply timeout to.</param>
        /// <param name="timeout">The maximum time allowed for the operation.</param>
        /// <param name="operationName">Name of the operation for error reporting.</param>
        /// <param name="ct">Optional cancellation token.</param>
        /// <returns>The result of the task if completed within the timeout period.</returns>
        /// <exception cref="TimeoutException">Thrown when the operation exceeds the specified timeout.</exception>
        public static async ValueTask<T> WithTimeout<T>(
            this ValueTask<T> task,
            TimeSpan timeout,
            string operationName = "operation",
            CancellationToken ct = default)
        {
            using var timeoutCts = new CancellationTokenSource(timeout);
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(timeoutCts.Token, ct);

            try
            {
                return await task.ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested)
            {
                throw new TimeoutException($"The {operationName} timed out after {timeout.TotalSeconds:F1} seconds");
            }
        }

        /// <summary>
        /// Executes an operation with automatic retry on failure.
        /// </summary>
        /// <typeparam name="T">The type of the operation result.</typeparam>
        /// <param name="operation">The operation to execute.</param>
        /// <param name="maxAttempts">Maximum number of retry attempts.</param>
        /// <param name="delay">Delay between retry attempts.</param>
        /// <param name="ct">Optional cancellation token.</param>
        /// <returns>The result of the operation if successful.</returns>
        /// <exception cref="AggregateException">Thrown when all retry attempts fail.</exception>
        public static async ValueTask<T> WithRetry<T>(
            this Func<CancellationToken, ValueTask<T>> operation,
            int maxAttempts = 3,
            TimeSpan? delay = null,
            CancellationToken ct = default)
        {
            delay ??= TimeSpan.FromSeconds(1);
            Exception lastException = null;

            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                try
                {
                    return await operation(ct).ConfigureAwait(false);
                }
                catch (Exception ex) when (attempt < maxAttempts - 1 &&
                                        !(ex is OperationCanceledException))
                {
                    lastException = ex;
                    await Task.Delay(delay.Value, ct).ConfigureAwait(false);
                }
            }

            throw new AggregateException($"Operation failed after {maxAttempts} attempts", lastException);
        }
    }
}