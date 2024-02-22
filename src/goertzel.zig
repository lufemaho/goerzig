const std = @import("std");

const PushError = error{ WrongSampleCount, OutputBufferTooSmall };
pub const Core = struct {
    allocator: std.mem.Allocator,
    samples_per_symbol: usize,
    phases: usize,
    buffer: []@Vector(4, f64),
    real_coeff: []@Vector(4, f64),
    imag_coeff: []@Vector(4, f64),
    core_count: usize,
    coeff: []@Vector(4, f64),
    state_before: []@Vector(4, f64),
    state_before_two: []@Vector(4, f64),

    /// Initialize a new goertzel core.
    /// allocator: is used the allocate memory for the internal state and goertzel coefficients.
    /// N: is the number of samples per symbol or analysis window.
    /// phases: the number of interleaved phases within one symbol
    pub fn init(allocator: std.mem.Allocator, N: usize, phases: usize, k: []const usize) !Core {
        const core_count = k.len * phases;
        const vec_count = if (core_count % 4 != 0)
            core_count / 4 + 1
        else
            core_count / 4;
        var buffer = try allocator.alignedAlloc(@Vector(4, f64), 32, 5 * vec_count);
        var core: Core = .{
            .allocator = allocator,
            .samples_per_symbol = N,
            .phases = phases,
            .buffer = buffer,
            .real_coeff = buffer[0 * vec_count .. 1 * vec_count],
            .imag_coeff = buffer[1 * vec_count .. 2 * vec_count],
            .core_count = phases * k.len,
            .coeff = buffer[2 * vec_count .. 3 * vec_count],
            .state_before = buffer[3 * vec_count .. 4 * vec_count],
            .state_before_two = buffer[4 * vec_count .. 5 * vec_count],
        };
        for (0..vec_count) |idx| {
            const N_f: @Vector(4, f64) = @splat(@as(f64, @floatFromInt(N)));
            const PI: @Vector(4, f64) = @splat(2.0 * std.math.pi);
            const curr_k = @Vector(4, f64){ @as(f64, @floatFromInt(k[(idx * 4 + 0) % k.len])), @as(f64, @floatFromInt(k[(idx * 4 + 1) % k.len])), @as(f64, @floatFromInt(k[(idx * 4 + 2) % k.len])), @as(f64, @floatFromInt(k[(idx * 4 + 3) % k.len])) };
            core.real_coeff[idx] = @cos(PI * curr_k / N_f);
            core.imag_coeff[idx] = @sin(PI * curr_k / N_f);
            core.coeff[idx] = @as(@Vector(4, f64), @splat(2.0)) * core.real_coeff[idx];
            core.state_before[idx] = @splat(0.0);
            core.state_before_two[idx] = @splat(0.0);
        }
        return core;
    }

    pub fn deinit(core: Core) void {
        core.allocator.free(core.buffer);
    }
    pub fn reset(self: Core) void {
        for (0..self.core_count) |idx| {
            self.state_before[idx] = @splat(0.0);
            self.state_before_two[idx] = @splat(0.0);
        }
    }

    /// Push samples for processing them and write the results into the otuput buffer.
    /// samples: slice of samples to process. If the size is not an integer multiple of
    /// the symbol size N, no samples are processed and PushError.WrongSampleCount is returned.
    /// output_buffer: a buffer that will contain the results. If the buffer is to small, no
    /// samples are processed and PushError.OutputBufferTooSmall is returned.
    pub fn push(self: Core, samples: []const i32, output_buffer: []f64) !usize {
        if (samples.len % self.samples_per_symbol != 0) {
            return PushError.WrongSampleCount;
        }
        const symbol_count = samples.len / self.samples_per_symbol;
        if (output_buffer.len < 2 * self.core_count * symbol_count) {
            return PushError.OutputBufferTooSmall;
        }
        const sub_slice_size = self.samples_per_symbol / self.phases;
        const total_phases_count = self.phases * symbol_count;
        const f_count = self.core_count / self.phases;
        for (0..total_phases_count) |result_idx| {
            const curr_slice = samples[result_idx * sub_slice_size .. (result_idx + 1) * sub_slice_size];
            const phase_idx = result_idx % self.phases;
            self.process(curr_slice);
            self.save_result(phase_idx, output_buffer.ptr + (result_idx * f_count * 2));
            self.reset_phase(phase_idx);
        }
        return self.core_count * symbol_count;
    }

    fn process(self: Core, samples: []const i32) void {
        @setRuntimeSafety(false);
        const vec_count = switch (self.core_count % 4) {
            0 => self.core_count / 4,
            else => self.core_count / 4 + 1,
        };
        for (samples) |curr_sample| {
            const sample_vec: @Vector(4, f64) = @splat(@as(f64, @floatFromInt(curr_sample)));
            for (0..vec_count) |vec_idx| {
                const state = sample_vec + self.coeff[vec_idx] * self.state_before[vec_idx] - self.state_before_two[vec_idx];
                self.state_before_two[vec_idx] = self.state_before[vec_idx];
                self.state_before[vec_idx] = state;
            }
        }
    }

    fn save_result(self: Core, phase_idx: usize, buffer: [*]f64) void {
        @setRuntimeSafety(false);
        const f_count = self.core_count / self.phases;
        for (0..f_count) |f_idx| {
            const curr_idx = phase_idx * f_count + f_idx;
            const vec_idx = curr_idx / 4;
            const sub_idx = curr_idx % 4;
            const real = self.state_before[vec_idx][sub_idx] * self.real_coeff[vec_idx][sub_idx] - self.state_before_two[vec_idx][sub_idx];
            const imag = self.state_before[vec_idx][sub_idx] * self.imag_coeff[vec_idx][sub_idx];
            buffer[2 * f_idx] = real;
            buffer[2 * f_idx + 1] = imag;
        }
    }

    fn reset_phase(self: Core, phase_idx: usize) void {
        @setRuntimeSafety(false);
        const f_count = self.core_count / self.phases;
        for (0..f_count) |f_idx| {
            const curr_idx = phase_idx * f_count + f_idx;
            const vec_idx = curr_idx / 4;
            const sub_idx = curr_idx % 4;
            self.state_before[vec_idx][sub_idx] = 0.0;
            self.state_before_two[vec_idx][sub_idx] = 0.0;
        }
    }
};

test "passing a single symbol to a single frequency and phase core returns one result" {
    const N = 10;
    const PHASE_COUNT = 1;
    const K: [1]usize = .{0};
    var core: Core = try Core.init(std.testing.allocator, N, PHASE_COUNT, &K);
    defer core.deinit();
    const buffer: [N]i32 = .{1} ++ .{0} ** 9;
    var output_buffer: [2]f64 = .{ -0.1, -0.2 };
    const res_count = try core.push(&buffer, &output_buffer);

    try std.testing.expectEqual(@as(usize, 1), res_count);
}

test "passing a single symbol to a single frequency and 8 phases core returns 8 results." {
    const N = 10;
    const PHASE_COUNT = 8;
    const K: [1]usize = .{0};
    var core: Core = try Core.init(std.testing.allocator, N, PHASE_COUNT, &K);
    defer core.deinit();
    const buffer: [N]i32 = .{1} ++ .{0} ** 9;
    var output_buffer: [2 * PHASE_COUNT]f64 = .{ -0.1, -0.2 } ** PHASE_COUNT;
    const res_count = try core.push(&buffer, &output_buffer);
    try std.testing.expectEqual(@as(usize, 8), res_count);
}

test "passing a single symbol to a 4 frequencies and 8 phases core returns 8 * 4 = 32 results." {
    const N = 10;
    const PHASE_COUNT = 8;
    const F_COUNT = 4;
    const K: [F_COUNT]usize = .{ 0, 1, 2, 3 };
    var core: Core = try Core.init(std.testing.allocator, N, PHASE_COUNT, &K);
    defer core.deinit();
    const buffer: [N]i32 = .{1} ++ .{0} ** 9;
    var output_buffer: [2 * PHASE_COUNT * F_COUNT]f64 = .{ -0.1, -0.2 } ** (PHASE_COUNT * F_COUNT);
    const res_count = try core.push(&buffer, &output_buffer);
    try std.testing.expectEqual(@as(usize, 32), res_count);
}

test "passing a sample buffer that is not an integer multiple size that of the cores symbol size returns an Error" {
    const N = 10;
    const PHASE_COUNT = 1;
    const K: [1]usize = .{0};
    var core: Core = try Core.init(std.testing.allocator, N, PHASE_COUNT, &K);
    defer core.deinit();
    const buffer: [N + 1]i32 = .{1} ++ .{0} ** N;
    var output_buffer: [2]f64 = .{ -0.1, -0.2 };
    try std.testing.expectError(PushError.WrongSampleCount, core.push(&buffer, &output_buffer));
}

test "passing a output buffer that is not big enough to hold all results returns an Error" {
    const N = 10;
    const PHASE_COUNT = 1;
    const K: [1]usize = .{0};
    var core: Core = try Core.init(std.testing.allocator, N, PHASE_COUNT, &K);
    defer core.deinit();
    const buffer: [N]i32 = .{1} ++ .{0} ** (N - 1);
    // output_buffer size must be at least 2:
    var output_buffer: [1]f64 = .{-0.1};
    try std.testing.expectError(PushError.OutputBufferTooSmall, core.push(&buffer, &output_buffer));
}

test "passing a dirac impulse returns that impulse for all frequencies" {
    const N = 10;
    const PHASE_COUNT = 1;
    const F_COUNT = 4;
    const K: [F_COUNT]usize = .{ 0, 1, 2, 5 };
    var core: Core = try Core.init(std.testing.allocator, N, PHASE_COUNT, &K);
    defer core.deinit();
    const buffer: [N]i32 = .{1} ++ .{0} ** 9;
    var output_buffer: [2 * PHASE_COUNT * F_COUNT]f64 = .{ -0.1, -0.2 } ** (PHASE_COUNT * F_COUNT);
    _ = try core.push(&buffer, &output_buffer);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), output_buffer[0], @as(f64, 1e-14));
    try std.testing.expectApproxEqRel(@as(f64, 1.0), output_buffer[2], @as(f64, 1e-14));
    try std.testing.expectApproxEqRel(@as(f64, 1.0), output_buffer[4], @as(f64, 1e-14));
    try std.testing.expectApproxEqRel(@as(f64, 1.0), output_buffer[6], @as(f64, 1e-14));
}

test "passing a cosine returns its amplitude as real part scaled by half the symbol size" {
    const N = 20;
    const PHASE_COUNT = 1;
    const K: [1]usize = .{1};
    const AMPLITUDE = 1e9;
    var core: Core = try Core.init(std.testing.allocator, N, PHASE_COUNT, &K);
    defer core.deinit();
    var buffer: [N]i32 = undefined;
    for (0..N) |idx| {
        buffer[idx] = @intFromFloat(AMPLITUDE * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(idx)) / @as(f64, @floatFromInt(N))));
    }
    var output_buffer: [2]f64 = .{ -0.1, -0.2 };
    _ = try core.push(&buffer, &output_buffer);
    try std.testing.expectApproxEqRel(@as(f64, AMPLITUDE * N) / 2.0, output_buffer[0], 1e-9);
}

test "passing a sine returns its amplitude as imaginary part scaled by half the symbol size" {
    const N = 20;
    const PHASE_COUNT = 1;
    const K: [1]usize = .{1};
    const AMPLITUDE = 1e9;
    var core: Core = try Core.init(std.testing.allocator, N, PHASE_COUNT, &K);
    defer core.deinit();
    var buffer: [N]i32 = undefined;
    for (0..N) |idx| {
        buffer[idx] = @intFromFloat(AMPLITUDE * @sin(2.0 * std.math.pi * @as(f64, @floatFromInt(idx)) / @as(f64, @floatFromInt(N))));
    }
    var output_buffer: [2]f64 = .{ -0.1, -0.2 };
    _ = try core.push(&buffer, &output_buffer);
    try std.testing.expectApproxEqRel(-@as(f64, AMPLITUDE * N) / 2.0, output_buffer[1], 1e-9);
}
