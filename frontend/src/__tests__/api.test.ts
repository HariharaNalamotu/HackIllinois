import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { submitTrainingJob, pollJobStatus, healthCheck, runInference } from '../services/api';
import type { PipelineSpec } from '../utils/pipelineBuilder';

// Mock SettingsModal exports
vi.mock('../components/SettingsModal', () => ({
  getStoredApiKey: () => 'test-key',
  getStoredBackendUrl: () => 'http://localhost:8787',
}));

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
});

describe('submitTrainingJob', () => {
  const spec: PipelineSpec = {
    pipeline_type: 'train',
    nodes: [
      { id: 'n1', type: 'text_input', params: {} },
      { id: 'n2', type: 'text_model', params: { base_model: 'all-MiniLM-L6-v2' } },
    ],
    edges: [{ from: 'n1', to: 'n2' }],
  };

  it('sends pipeline spec and files as FormData to correct URL', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ job_id: 'job-123' }),
    });

    const file = new File(['hello world'], 'test.txt', { type: 'text/plain' });
    const result = await submitTrainingJob('wf-abc', spec, [{ nodeId: 'n1', file }]);

    expect(result.job_id).toBe('job-123');
    expect(mockFetch).toHaveBeenCalledOnce();

    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toBe('http://localhost:8787/api/workflow/wf-abc/train');
    expect(opts.method).toBe('POST');

    // Verify FormData contents
    const fd = opts.body as FormData;
    expect(fd.get('pipeline')).toBe(JSON.stringify(spec));

    // File should be keyed as files[n1]
    const sentFile = fd.get('files[n1]');
    expect(sentFile).toBeInstanceOf(File);
    expect((sentFile as File).name).toBe('test.txt');
  });

  it('sends files keyed by nodeId — critical for text_chunks routing', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ job_id: 'job-456' }),
    });

    const f1 = new File(['data1'], 'a.pdf', { type: 'application/pdf' });
    const f2 = new File(['data2'], 'b.txt', { type: 'text/plain' });

    await submitTrainingJob('wf-x', spec, [
      { nodeId: 'node-aaa', file: f1 },
      { nodeId: 'node-aaa', file: f2 },
    ]);

    const fd = mockFetch.mock.calls[0][1].body as FormData;
    // Both files appended under same nodeId key
    const allEntries = fd.getAll('files[node-aaa]');
    expect(allEntries).toHaveLength(2);
  });

  it('throws on HTTP error with detail', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      text: async () => '{"error":"Modal /train returned 500"}',
    });

    await expect(submitTrainingJob('wf-x', spec, [])).rejects.toThrow(
      /Training submission failed \(HTTP 502\)/
    );
  });

  it('sends zero files when collectFiles returns empty — the silent failure case', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ job_id: 'job-789' }),
    });

    // Simulating what happens when uploadedFiles is a string (the old bug)
    await submitTrainingJob('wf-x', spec, []);

    const fd = mockFetch.mock.calls[0][1].body as FormData;
    // No files[*] entries at all — this is what caused the TextInputNode error
    const allKeys = Array.from((fd as any).keys());
    const fileKeys = allKeys.filter((k: string) => k.startsWith('files['));
    expect(fileKeys).toHaveLength(0);
  });
});

describe('pollJobStatus', () => {
  it('returns job status on success', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: 'complete', result: { output_name: 'model-x' } }),
    });
    const result = await pollJobStatus('job-123');
    expect(result.status).toBe('complete');
    expect(mockFetch.mock.calls[0][0]).toBe('http://localhost:8787/api/pipeline/job-123/status');
  });

  it('returns not_found on HTTP error', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });
    const result = await pollJobStatus('bad-id');
    expect(result.status).toBe('not_found');
  });
});

describe('healthCheck', () => {
  it('returns true when backend is up', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true });
    expect(await healthCheck()).toBe(true);
  });

  it('returns false when backend is down', async () => {
    mockFetch.mockRejectedValueOnce(new Error('ECONNREFUSED'));
    expect(await healthCheck()).toBe(false);
  });
});

describe('runInference', () => {
  it('sends text input as plain text', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ result: { output: 'hello' } }),
    });

    await runInference('wf-1', 'some input text');

    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toBe('http://localhost:8787/api/deploy/wf-1');
    expect(opts.headers['Content-Type']).toBe('text/plain');
    expect(opts.body).toBe('some input text');
  });

  it('sends File input as FormData', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ result: {} }),
    });

    const file = new File(['img'], 'photo.jpg', { type: 'image/jpeg' });
    await runInference('wf-1', file);

    const fd = mockFetch.mock.calls[0][1].body as FormData;
    expect(fd.get('file')).toBeInstanceOf(File);
  });

  it('throws on inference failure', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      text: async () => 'Workflow not found',
    });

    await expect(runInference('bad-id', 'test')).rejects.toThrow(/Inference failed/);
  });
});
