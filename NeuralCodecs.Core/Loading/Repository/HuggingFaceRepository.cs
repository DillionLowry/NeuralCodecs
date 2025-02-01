using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Utils;
using System.Text.Json;

namespace NeuralCodecs.Core.Loading.Repository
{
    /// <summary>
    /// Implements the model repository interface for the Hugging Face model hub.
    /// Provides functionality to download and manage machine learning models from huggingface.co.
    /// </summary>
    public class HuggingFaceRepository : IModelRepository, IDisposable
    {
        private readonly HttpClient _client;
        private const string API_BASE = "https://huggingface.co/api";
        private const string REPO_BASE = "https://huggingface.co";
        private const int DEFAULT_TIMEOUT_SECONDS = 300;
        private bool disposedValue;

        /// <summary>
        /// Initializes a new instance of the HuggingFaceRepository class.
        /// </summary>
        /// <param name="authToken">Optional authentication token for accessing private repositories.</param>
        public HuggingFaceRepository(string? authToken = null)
        {
            _client = CreateHttpClient(authToken);
        }

        private HttpClient CreateHttpClient(string? authToken)
        {
            var client = new HttpClient
            {
                BaseAddress = new Uri(REPO_BASE),
                Timeout = TimeSpan.FromSeconds(DEFAULT_TIMEOUT_SECONDS)
            };

            client.DefaultRequestHeaders.Add("User-Agent", "NeuralCodecs/1.0");

            if (!string.IsNullOrEmpty(authToken))
            {
                client.DefaultRequestHeaders.Authorization =
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", authToken);
            }

            return client;
        }

        /// <summary>
        /// Retrieves the path of the main model file from the specified repository.
        /// </summary>
        /// <param name="modelId">The Hugging Face model identifier.</param>
        /// <param name="revision">The repository revision or branch name.</param>
        /// <returns>The path to the main model file within the repository.</returns>
        /// <exception cref="LoadException">Thrown when the model file cannot be found or accessed.</exception>
        public async Task<string> GetModelPath(string modelId, string revision)
        {
            try
            {
                var files = await GetRepositoryFiles(modelId, revision);
                var modelFile = files.FirstOrDefault(f =>
                    Path.GetExtension(f.Path) is ".bin" or ".pt" or ".pth");

                if (modelFile == null)
                {
                    throw new LoadException($"No model file found in repository: {modelId}");
                }

                return modelFile.Path;
            }
            catch (Exception ex)
            {
                throw new LoadException($"Failed to get model path for {modelId}", ex);
            }
        }

        /// <summary>
        /// Retrieves metadata information about a model from the repository.
        /// </summary>
        /// <param name="modelId">The Hugging Face model identifier.</param>
        /// <returns>A ModelMetadata object containing information about the model.</returns>
        /// <exception cref="LoadException">Thrown when the model information cannot be retrieved.</exception>
        public async Task<ModelMetadata> GetModelInfo(string modelId)
        {
            try
            {
                var metadata = await GetRepositoryMetadata(modelId);
                // Find model file
                var modelFile = metadata.Files.FirstOrDefault(f => FileUtil.IsValidModelFile(f.Path));
            
                if (modelFile == null)
                {
                    throw new LoadException($"No model file found in repository: {modelId}");
                }

                // Find config file using multiple patterns
                var configFile = metadata.Files.FirstOrDefault(f =>
                {
                    var fileName = f.Path.ToLowerInvariant();
                    var modelName = Path.GetFileNameWithoutExtension(modelFile.Path).ToLowerInvariant();

                    return fileName == "config.json" ||                    // Standard config
                           fileName == $"{modelName}.json" ||             // Model-specific config
                           fileName == $"{modelName}_config.json" ||      // Alternative format
                           fileName.EndsWith("_config.json") ||           // Generic config
                           (fileName.Contains("config") &&
                            fileName.EndsWith(".json"));                  // Any config file
                });

                return new ModelMetadata
                {
                    Source = modelId,
                    LastModified = metadata.LastModified,
                    Author = metadata.Author,
                    Tags = metadata.Tags,
                    Backend = "TorchSharp",
                    Size = modelFile?.Size ?? 0,
                    FileName = modelFile?.Path ?? "",
                    ConfigFileName = configFile?.Path ?? ""
                };
            }
            catch (Exception ex)
            {
                throw new LoadException($"Failed to get model info for {modelId}", ex);
            }
        }

        /// <summary>
        /// Downloads a model and its associated files from the repository.
        /// </summary>
        /// <param name="modelId">The Hugging Face model identifier.</param>
        /// <param name="targetPath">The local directory path where the model should be downloaded.</param>
        /// <param name="progress">An IProgress object to report download progress.</param>
        /// <exception cref="LoadException">Thrown when the download fails or validation errors occur.</exception>
        public async Task DownloadModel(string modelId, string targetPath, IProgress<double> progress, ModelLoadOptions options)
        {

            try
            {
                var files = await GetRepositoryFiles(modelId, "main");
                var exts = ModelFileTypeExtensions.GetDefaultExtensions();
                var filesToDownload = files.Where(f => exts.Any(ext => f.Path.EndsWith(ext))).ToList();
                var totalSize = filesToDownload.Sum(f => f.Size);
                var downloadedSize = 0L;
                foreach (var file in filesToDownload)
                {
                    var destPath = Path.Combine(targetPath, file.Path);
                    var destDir = Path.GetDirectoryName(destPath);
                    if (destDir != null) Directory.CreateDirectory(destDir);

                    await DownloadFile(modelId, file, destPath, new Progress<long>(bytesDownloaded =>
                    {
                        downloadedSize += bytesDownloaded;
                        progress.Report((double)downloadedSize / totalSize);
                    }));
                }
                
                await ValidateDownload(targetPath, filesToDownload, options);
            }
            catch (Exception ex)
            {
                // Cleanup on failure
                try
                {
                    if (Directory.Exists(targetPath))
                    {
                        Directory.Delete(targetPath, recursive: true);
                    }
                }
                catch { /* Ignore cleanup errors */ }

                throw new LoadException($"Failed to download model {modelId}", ex);
            }
        }

        private async Task<List<RepositoryFile>> GetRepositoryFiles(string modelId, string revision)
        {
            try
            {
                var response = await _client.GetAsync($"{API_BASE}/models/{modelId}/tree/{revision}");
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync();
                var files = JsonSerializer.Deserialize<List<RepositoryFile>>(json,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                return files ?? throw new LoadException("Failed to get repository file listing");
            }
            catch (Exception ex)
            {
                throw new LoadException($"Failed to get file listing for {modelId}", ex);
            }
        }

        private async Task<RepositoryMetadata> GetRepositoryMetadata(string modelId)
        {
            try
            {
                var response = await _client.GetAsync($"{API_BASE}/models/{modelId}");
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync();
                var metadata = JsonSerializer.Deserialize<RepositoryMetadata>(json,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (metadata is null)
                {
                    throw new LoadException("Failed to get repository metadata");
                }
                metadata.Files = await GetRepositoryFiles(modelId, "main");
                return metadata; 
            }
            catch (Exception ex) when (ex is not LoadException)
            {
                throw new LoadException($"Failed to get metadata for {modelId}", ex);
            }
        }

        private async Task DownloadFile(
            string modelId,
            RepositoryFile file,
            string destPath,
            IProgress<long> progress,
            CancellationToken cancellationToken = default)
        {
            var tempPath = destPath + ".tmp";

            try
            {
                var url = $"{modelId}/resolve/main/{file.Path}";
                using var response = await _client.GetAsync(
                    url,
                    HttpCompletionOption.ResponseHeadersRead,
                    cancellationToken);
                response.EnsureSuccessStatusCode();

                await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
                await using var fileStream = new FileStream(
                    tempPath,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.None,
                    bufferSize: 81920,
                    useAsync: true);

                await contentStream.CopyToAsync(fileStream, progress, cancellationToken);

                // Verify file size if known
                if (file.Size > 0 && fileStream.Length != file.Size)
                {
                    throw new LoadException(
                        $"Downloaded file size mismatch for {file.Path}. " +
                        $"Expected {file.Size} bytes but got {fileStream.Length}");
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);

                throw new LoadException($"Failed to download {file.Path}", ex);
            }

            try
            {
                // Atomic move
                if (File.Exists(destPath))
                    File.Delete(destPath);
                File.Move(tempPath, destPath);
            }
            catch (Exception ex)
            {
                throw new LoadException($"Failed to move downloaded file {file.Path}", ex);
            }
        }

        private async Task ValidateDownload(string targetDir, List<RepositoryFile> expectedFiles, ModelLoadOptions options)
        {
            // Verify all required files exist
            var missingFiles = expectedFiles.Where(f =>
                !File.Exists(Path.Combine(targetDir, f.Path))).ToList();

            if (missingFiles.Count != 0)
            {
                throw new LoadException(
                    $"Missing files after download: {string.Join(", ", missingFiles.Select(f => f.Path))}");
            }

            // Verify model file exists
            var modelFile = expectedFiles.FirstOrDefault(f => FileUtil.IsValidModelFile(f.Path));

            if (modelFile == null)
            {
                throw new LoadException("No model file found in downloaded content");
            }

            var modelPath = Path.Combine(targetDir, modelFile.Path);

            if (options.HasConfigFile)
            {
                // Verify config exists
                var configPath = Path.Combine(targetDir, "config.json");
                if (!File.Exists(configPath))
                {
                    configPath = Path.ChangeExtension(modelPath, ".json");
                    if (!File.Exists(configPath))
                    {
                        throw new LoadException("Config file missing from download");
                    }
                }
            }

            // Verify file sizes
            foreach (var file in expectedFiles)
            {
                var filePath = Path.Combine(targetDir, file.Path);
                var fileInfo = new FileInfo(filePath);

                if (fileInfo.Length != file.Size)
                {
                    throw new LoadException(
                        $"File size mismatch for {file.Path}. " +
                        $"Expected {file.Size} bytes but got {fileInfo.Length}");
                }
            }

            // Verify model file format
            if (options.ValidateModel)
            {
                var fileType = await FileUtil.DetectFileTypeFromContentsAsync(modelPath);

                if (fileType is ModelFileType.Unknown)
                {
                    throw new LoadException("Failed to validate model file type");
                }
            }
        }


        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    _client.Dispose();
                }
                disposedValue = true;
            }
        }
        
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}