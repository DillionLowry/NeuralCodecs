using NeuralCodecs.Core.Exceptions;
using NeuralCodecs.Core.Interfaces;
using NeuralCodecs.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace NeuralCodecs.Core.Loading
{
	public class HuggingFaceRepository : IModelRepository, IDisposable
	{
		private readonly HttpClient _client;
		private const string API_BASE = "https://huggingface.co/api";
		private const string REPO_BASE = "https://huggingface.co";
		private const int DEFAULT_TIMEOUT_SECONDS = 300;
		private readonly string[] _allowedFilePatterns = new[] { "*.bin", "*.pt", "*.pth", "*.json" };

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

		public async Task<string> GetModelPath(string modelId, string revision)
		{
			try
			{
				var files = await GetRepositoryFiles(modelId, revision);
				var modelFile = files.FirstOrDefault(f =>
					Path.GetExtension(f.Path) is ".bin" or ".pt" or ".pth");

				if (modelFile == null)
				{
					throw new ModelLoadException($"No model file found in repository: {modelId}");
				}

				return modelFile.Path;
			}
			catch (Exception ex)
			{
				throw new ModelLoadException($"Failed to get model path for {modelId}", ex);
			}
		}

		public async Task<ModelInfo> GetModelInfo(string modelId)
		{
			try
			{
				var metadata = await GetRepositoryMetadata(modelId);
				var modelFile = metadata.Files.FirstOrDefault(f =>
					Path.GetExtension(f.Path) is ".bin" or ".pt" or ".pth");

				return new ModelInfo
				{
					Source = modelId,
					LastModified = metadata.LastModified,
					Author = metadata.Author,
					Tags = metadata.Tags,
					Backend = "TorchSharp",
					Size = modelFile?.Size ?? 0
				};
			}
			catch (Exception ex)
			{
				throw new ModelLoadException($"Failed to get model info for {modelId}", ex);
			}
		}

		public async Task DownloadModel(string modelId, string targetPath, IProgress<double> progress)
		{
			var targetDir = Path.GetDirectoryName(targetPath)
				?? throw new ArgumentException("Invalid target path", nameof(targetPath));

			Directory.CreateDirectory(targetDir);

			try
			{
				var files = await GetRepositoryFiles(modelId, "main");
				var filesToDownload = files.Where(f =>
					_allowedFilePatterns.Any(p =>
						new WildcardPattern(p).IsMatch(f.Path))).ToList();

				var totalSize = filesToDownload.Sum(f => f.Size);
				var downloadedSize = 0L;

				foreach (var file in filesToDownload)
				{
					var destPath = Path.Combine(targetDir, file.Path);
					var destDir = Path.GetDirectoryName(destPath);
					if (destDir != null) Directory.CreateDirectory(destDir);

					await DownloadFile(modelId, file, destPath, new Progress<long>(bytesDownloaded =>
					{
						downloadedSize += bytesDownloaded;
						progress.Report((double)downloadedSize / totalSize);
					}));
				}

				// Validate downloaded files
				await ValidateDownload(targetDir, filesToDownload);
			}
			catch (Exception ex)
			{
				// Cleanup on failure
				try
				{
					if (Directory.Exists(targetDir))
					{
						Directory.Delete(targetDir, recursive: true);
					}
				}
				catch
				{
					// Ignore cleanup errors
				}

				throw new ModelLoadException($"Failed to download model {modelId}", ex);
			}
		}

		private async Task<List<RepoFile>> GetRepositoryFiles(string modelId, string revision)
		{
			try
			{
				var response = await _client.GetAsync($"{API_BASE}/models/{modelId}/tree/{revision}");
				response.EnsureSuccessStatusCode();

				var json = await response.Content.ReadAsStringAsync();
				var files = JsonSerializer.Deserialize<List<RepoFile>>(json,
					new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

				return files ?? throw new ModelLoadException("Failed to get repository file listing");
			}
			catch (Exception ex)
			{
				throw new ModelLoadException($"Failed to get file listing for {modelId}", ex);
			}
		}

		private async Task<RepoMetadata> GetRepositoryMetadata(string modelId)
		{
			try
			{
				var response = await _client.GetAsync($"{API_BASE}/models/{modelId}");
				response.EnsureSuccessStatusCode();

				var json = await response.Content.ReadAsStringAsync();
				var metadata = JsonSerializer.Deserialize<RepoMetadata>(json,
					new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

				return metadata ?? throw new ModelLoadException("Failed to get repository metadata");
			}
			catch (Exception ex)
			{
				throw new ModelLoadException($"Failed to get metadata for {modelId}", ex);
			}
		}

		private async Task DownloadFile(
			string modelId,
			RepoFile file,
			string destPath,
			IProgress<long> progress)
		{
			var tempPath = destPath + ".tmp";

			try
			{
				var url = $"{modelId}/resolve/main/{file.Path}";
				using var response = await _client.GetAsync(
					url,
					HttpCompletionOption.ResponseHeadersRead);
				response.EnsureSuccessStatusCode();

				using var contentStream = await response.Content.ReadAsStreamAsync();
				using var fileStream = new FileStream(
					tempPath,
					FileMode.Create,
					FileAccess.Write,
					FileShare.None,
					bufferSize: 81920,
					useAsync: true);

				var buffer = new byte[81920];
				int bytesRead;
				while ((bytesRead = await contentStream.ReadAsync(buffer)) != 0)
				{
					await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead));
					progress.Report(bytesRead);
				}

				// Verify file size if known
				if (file.Size > 0 && fileStream.Length != file.Size)
				{
					throw new ModelLoadException(
						$"Downloaded file size mismatch for {file.Path}. " +
						$"Expected {file.Size} bytes but got {fileStream.Length}");
				}
			}
			catch (Exception ex)
			{
				if (File.Exists(tempPath))
					File.Delete(tempPath);

				throw new ModelLoadException($"Failed to download {file.Path}", ex);
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
				throw new ModelLoadException($"Failed to move downloaded file {file.Path}", ex);
			}
		}

		private async Task ValidateDownload(string targetDir, List<RepoFile> expectedFiles)
		{
			// Verify all required files exist
			var missingFiles = expectedFiles.Where(f =>
				!File.Exists(Path.Combine(targetDir, f.Path))).ToList();

			if (missingFiles.Any())
			{
				throw new ModelLoadException(
					$"Missing files after download: {string.Join(", ", missingFiles.Select(f => f.Path))}");
			}

			// Verify model file exists
			var modelFile = expectedFiles.FirstOrDefault(f =>
				Path.GetExtension(f.Path) is ".bin" or ".pt" or ".pth");

			if (modelFile == null)
			{
				throw new ModelLoadException("No model file found in downloaded content");
			}

			var modelPath = Path.Combine(targetDir, modelFile.Path);

			// Verify config exists
			var configPath = Path.ChangeExtension(modelPath, ".json");
			if (!File.Exists(configPath))
			{
				configPath = Path.Combine(targetDir, "config.json");
				if (!File.Exists(configPath))
				{
					throw new ModelLoadException("Config file missing from download");
				}
			}

			// Verify file sizes
			foreach (var file in expectedFiles)
			{
				var filePath = Path.Combine(targetDir, file.Path);
				var fileInfo = new FileInfo(filePath);

				if (fileInfo.Length != file.Size)
				{
					throw new ModelLoadException(
						$"File size mismatch for {file.Path}. " +
						$"Expected {file.Size} bytes but got {fileInfo.Length}");
				}
			}

			// Verify model file format
			try
			{
				var modelBytes = await File.ReadAllBytesAsync(modelPath);
				if (!IsPyTorchModelFile(modelBytes))
				{
					throw new ModelLoadException("Invalid PyTorch model file format");
				}
			}
			catch (Exception ex)
			{
				throw new ModelLoadException("Failed to validate model file", ex);
			}
		}

		private bool IsPyTorchModelFile(byte[] bytes)
		{
			if (bytes.Length < 8) return false;

			// Look for common PyTorch model signatures
			return (bytes[0] == 0x80 && bytes[1] == 0x02) || // Pickle protocol
				   (bytes[0] == 'P' && bytes[1] == 'K');     // ZIP archive
		}

		public void Dispose()
		{
			_client.Dispose();
		}
	}
}
