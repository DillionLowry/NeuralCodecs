using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Utils;
using System.Net.Http.Headers;
using System.Text.Json;

namespace NeuralCodecs.Core.Loading.Repository
{
    /// <summary>
    /// Implements the model repository interface for direct URL downloads.
    /// Provides functionality to download models from any direct HTTP URL.
    /// </summary>
    public class DirectUrlRepository : IModelRepository, IDisposable
    {
        private readonly HttpClient _client;
        private const int DEFAULT_TIMEOUT_SECONDS = 300;
        private readonly string[] _supportedDomains;
        private bool disposedValue;

        /// <summary>
        /// Initializes a new instance of the DirectUrlRepository class.
        /// </summary>
        /// <param name="supportedDomains">Optional array of domain names this repository supports. 
        /// If null or empty, all domains are supported.</param>
        public DirectUrlRepository(string[]? supportedDomains = null)
        {
            _client = CreateHttpClient();
            _supportedDomains = supportedDomains ?? Array.Empty<string>();
        }

        private HttpClient CreateHttpClient()
        {
            var client = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(DEFAULT_TIMEOUT_SECONDS)
            };

            client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("*/*"));
            client.DefaultRequestHeaders.Add("User-Agent", "NeuralCodecs/1.0");

            return client;
        }

        /// <summary>
        /// Determines if this repository can handle the specified URL.
        /// </summary>
        /// <param name="url">The URL to check.</param>
        /// <returns>True if this repository can handle the URL; otherwise, false.</returns>
        public bool CanHandleUrl(string url)
        {
            if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
                return false;
                
            if (!uri.Scheme.Equals("http", StringComparison.OrdinalIgnoreCase) && 
                !uri.Scheme.Equals("https", StringComparison.OrdinalIgnoreCase))
                return false;

            // If no supported domains are specified, accept all domains
            if (_supportedDomains.Length == 0)
                return true;
                
            // Check if the URL's domain is in the supported domains list
            return _supportedDomains.Any(domain => 
                uri.Host.Equals(domain, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Gets the path to a model file within the repository.
        /// </summary>
        /// <param name="url">The URL to the model file.</param>
        /// <param name="revision">Ignored for this repository type.</param>
        /// <returns>The filename portion of the URL path.</returns>
        public Task<string> GetModelPath(string url, string revision)
        {
            if (!CanHandleUrl(url))
            {
                throw new LoadException($"Invalid URL or unsupported domain: {url}");
            }

            // Extract the filename from the URL
            var uri = new Uri(url);
            var filename = Path.GetFileName(uri.AbsolutePath);
            
            if (string.IsNullOrEmpty(filename))
            {
                throw new LoadException($"Could not determine filename from URL: {url}");
            }
            
            return Task.FromResult(filename);
        }

        /// <summary>
        /// Retrieves metadata information about a model.
        /// </summary>
        /// <param name="url">The URL to the model file.</param>
        /// <param name="revision">Ignored for this repository type.</param>
        /// <returns>Basic model metadata based on the URL.</returns>
        public async Task<ModelMetadata> GetModelInfo(string url, string revision)
        {
            if (!CanHandleUrl(url))
            {
                throw new LoadException($"Invalid URL or unsupported domain: {url}");
            }

            var uri = new Uri(url);
            var filename = Path.GetFileName(uri.AbsolutePath);
            var configFilename = Path.ChangeExtension(filename, ".json");

            // Get the directory part of the URL
            var urlDir = uri.AbsolutePath.Substring(0, uri.AbsolutePath.Length - filename.Length).TrimEnd('/');

            // Check if config exists by trying to access it
            var configUrl = $"{urlDir}/{configFilename}";
            var hasConfig = await DoesFileExist(configUrl);

            // Try to get file size via HEAD request
            long fileSize = 0;
            try
            {
                var response = await _client.SendAsync(new HttpRequestMessage(HttpMethod.Head, url));
                if (response.IsSuccessStatusCode && response.Content.Headers.ContentLength.HasValue)
                {
                    fileSize = response.Content.Headers.ContentLength.Value;
                }
            }
            catch
            {
                // Ignore errors during size check
            }

            return new ModelMetadata
            {
                Source = urlDir,
                FileName = filename,
                ConfigFileName = hasConfig ? configFilename : "",
                Size = fileSize,
                LastModified = DateTime.UtcNow, // Actual timestamp not available
                Author = "Unknown", // No author info for direct URLs
                Backend = "TorchSharp",
                Tags = new List<string> { "direct-url" }
            };
        }

        /// <summary>
        /// Downloads a model from the repository.
        /// </summary>
        /// <param name="url">The URL to the model.</param>
        /// <param name="targetPath">The local path where the model should be saved.</param>
        /// <param name="progress">A progress reporter for download status.</param>
        /// <param name="options">Model loading options.</param>
        public async Task DownloadModel(string url, string targetPath, IProgress<double> progress, ModelLoadOptions options)
        {
            if (!CanHandleUrl(url))
            {
                throw new LoadException($"Invalid URL or unsupported domain: {url}");
            }

            try
            {
                var uri = new Uri(url);
                var filename = Path.GetFileName(uri.AbsolutePath);
                var modelFilePath = Path.Combine(targetPath, filename);
                
                // Create target directory
                Directory.CreateDirectory(targetPath);
                
                // Download the model file
                await DownloadFile(url, modelFilePath, progress);
                
                // Try to download a matching config file if requested
                if (options.HasConfigFile)
                {
                    // Get the directory part of the URL
                    var urlDir = url.Substring(0, url.Length - filename.Length);
                    if (urlDir.EndsWith("/"))
                        urlDir = urlDir.Substring(0, urlDir.Length - 1);
                        
                    var configFilename = Path.ChangeExtension(filename, ".json");
                    var configUrl = $"{urlDir}/{configFilename}";
                    var configFilePath = Path.Combine(targetPath, configFilename);
                    
                    if (await DoesFileExist(configUrl))
                    {
                        await DownloadFile(configUrl, configFilePath, null);
                    }
                }
                
                await ValidateDownload(targetPath, filename, options);
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

                throw new LoadException($"Failed to download model from {url}", ex);
            }
        }
        
        private async Task<bool> DoesFileExist(string url)
        {
            try
            {
                var response = await _client.SendAsync(new HttpRequestMessage(HttpMethod.Head, url));
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        private async Task DownloadFile(string url, string destPath, IProgress<double>? progress)
        {
            var tempPath = destPath + ".tmp";

            try
            {
                using var response = await _client.GetAsync(
                    url,
                    HttpCompletionOption.ResponseHeadersRead);
                response.EnsureSuccessStatusCode();

                var totalBytes = response.Content.Headers.ContentLength ?? -1L;

                await using var contentStream = await response.Content.ReadAsStreamAsync();
                await using var fileStream = new FileStream(
                    tempPath,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.None,
                    bufferSize: 81920,
                    useAsync: true);

                if (progress != null && totalBytes > 0)
                {
                    await contentStream.CopyToAsync(fileStream, 
                        new Progress<long>(bytesRead => progress.Report((double)bytesRead / totalBytes)));
                }
                else
                {
                    await contentStream.CopyToAsync(fileStream);
                }
            }
            catch (Exception ex)
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);

                throw new LoadException($"Failed to download file from {url}", ex);
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
                throw new LoadException($"Failed to move downloaded file to {destPath}", ex);
            }
        }

        private async Task ValidateDownload(string targetPath, string filename, ModelLoadOptions options)
        {
            var modelPath = Path.Combine(targetPath, filename);
            
            if (!File.Exists(modelPath))
            {
                throw new LoadException($"Downloaded model file not found at {modelPath}");
            }

            if (options.HasConfigFile)
            {
                var configPath = Path.Combine(targetPath, Path.ChangeExtension(filename, ".json"));
                if (!File.Exists(configPath))
                {
                    configPath = Path.Combine(targetPath, "config.json");
                    if (!File.Exists(configPath))
                    {
                        // Config is optional for direct downloads
                        Console.WriteLine("No configuration file found for the downloaded model");
                    }
                }
            }

            if (options.ValidateModel)
            {
                var fileType = await FileUtils.DetectFileTypeFromContentsAsync(modelPath);
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
