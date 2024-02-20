const std = @import("std");
const clap = @import("zig-clap/clap.zig");
const InputError = error{ NotEnoughArguments, PhasesValueInvalid, BufferSizeTooSmall };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const settings = try parse_cli_args(gpa.allocator());
    defer settings.deinit();
    if (settings.help_only) return;
    var core = try Core.init(gpa.allocator(), settings.symbol_size, settings.phases, settings.k);
    defer core.deinit();

    // file input buffer
    var path_buffer: [std.fs.MAX_PATH_BYTES]u8 = undefined;
    const inputfile = if (settings.file) |file| try std.fs.openFileAbsolute(try std.fs.realpath(file, &path_buffer), .{}) else std.io.getStdIn();
    defer inputfile.close();
    const max_buff_size = settings.buffer_size;
    const buffer_size = max_buff_size - (max_buff_size % settings.symbol_size);
    debug_print("Max. buffer size: {}\nActual buffer size: {}", .{ max_buff_size, buffer_size });
    var buffer = try gpa.allocator().alloc(i32, buffer_size);
    defer gpa.allocator().free(buffer);
    var bytes_written: usize = 0;

    // file output buffer
    const stdout_file = std.io.getStdOut().writer();
    var writer = std.io.bufferedWriter(stdout_file);
    var out_buffer = try gpa.allocator().alignedAlloc(f64, 32, 2 * settings.k.len * settings.phases * buffer_size / settings.symbol_size);
    defer gpa.allocator().free(out_buffer);

    while (true) {
        debug_print("pushing buffer", .{});
        const bytes_read = try inputfile.readAll(std.mem.sliceAsBytes(buffer));
        const samples_read = bytes_read / @sizeOf(i32);
        if (bytes_read > 0) {
            const result_count = try core.push(buffer[0..samples_read], out_buffer);
            bytes_written += try writer.write(std.mem.sliceAsBytes(out_buffer[0 .. 2 * result_count]));
            try writer.flush(); // don't forget to flush!
            debug_print("out buffer: {any}", .{out_buffer}); //
        } else {
            break;
        }
    }
    try std.fmt.format(std.io.getStdErr().writer(), "\nbytes written: {d}\n", .{bytes_written});
}
const PushError = error{ WrongSampleCount, OutputBufferTooSmall };
const Core = struct {
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
            debug_print("real coeff: {}, imag coeff: {}", .{ core.real_coeff[idx], core.imag_coeff[idx] });
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
        for (0..total_phases_count) |result_idx| {
            const curr_slice = samples[result_idx * sub_slice_size .. (result_idx + 1) * sub_slice_size];
            const phase_idx = result_idx % self.phases;
            self.process(curr_slice);
            self.save_result(phase_idx, output_buffer.ptr + (result_idx * self.phases * 2));
            self.reset_phase(phase_idx);
        }
        return self.core_count * symbol_count;
    }

    fn process(self: Core, samples: []const i32) void {
        const vec_count = self.core_count / 4;
        for (samples) |curr_sample| {
            const sample_vec: @Vector(4, f64) = @splat(@as(f64, @floatFromInt(curr_sample)));
            for (0..vec_count) |vec_idx| {
                const state = sample_vec + self.coeff[vec_idx] * self.state_before[vec_idx] - self.state_before_two[vec_idx];
                // std.debug.print("{}, ", .{state});
                self.state_before_two[vec_idx] = self.state_before[vec_idx];
                self.state_before[vec_idx] = state;
            }
        }
    }

    fn save_result(self: Core, phase_idx: usize, buffer: [*]f64) void {
        const f_count = self.core_count / self.phases;
        for (0..f_count) |f_idx| {
            const curr_idx = phase_idx * f_count + f_idx;
            const vec_idx = curr_idx / 4;
            const sub_idx = curr_idx % 4;
            const real = self.state_before[vec_idx][sub_idx] * self.real_coeff[vec_idx][sub_idx] - self.state_before_two[vec_idx][sub_idx];
            const imag = self.state_before[vec_idx][sub_idx] * self.imag_coeff[vec_idx][sub_idx];
            buffer[2 * curr_idx] = real;
            buffer[2 * curr_idx + 1] = imag;
            debug_print("{}, {}, {}", .{ curr_idx, real, imag });
        }
    }

    fn reset_phase(self: Core, phase_idx: usize) void {
        const f_count = self.core_count / self.phases;
        for (0..f_count) |f_idx| {
            const curr_idx = phase_idx * f_count + f_idx;
            const vec_idx = curr_idx / 4;
            const sub_idx = curr_idx % 4;
            debug_print("reseting core: {}", .{curr_idx});
            self.state_before[vec_idx][sub_idx] = 0.0;
            self.state_before_two[vec_idx][sub_idx] = 0.0;
        }
    }
    // pub fn format(self: Core, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
    //     _ = writer;
    //     _ = options;
    //     _ = fmt;
    //     _ = self;
    // }
};
const CLIArgs = struct {
    allocator: std.mem.Allocator,
    phases: usize,
    symbol_size: usize,
    k: []const usize,
    buffer_size: usize,
    help_only: bool,
    file: ?[]const u8,

    fn init(
        allocator: std.mem.Allocator,
        phases: usize,
        symbol_size: usize,
        k: []const usize,
        buffer_size: usize,
        help_only: bool,
        file: ?[]const u8,
    ) !CLIArgs {
        var k_buff = try allocator.alloc(usize, k.len);
        for (0..k.len) |i| {
            k_buff[i] = k[i];
        }
        return CLIArgs{
            .file = file,
            .help_only = help_only,
            .buffer_size = buffer_size,
            .k = k_buff,
            .symbol_size = symbol_size,
            .phases = phases,
            .allocator = allocator,
        };
    }

    fn init_help_only() !CLIArgs {
        return CLIArgs{
            .file = undefined,
            .help_only = true,
            .buffer_size = undefined,
            .k = undefined,
            .symbol_size = undefined,
            .phases = undefined,
            .allocator = undefined,
        };
    }
    fn deinit(self: CLIArgs) void {
        if (!self.help_only)
            self.allocator.free(self.k);
    }
};
fn parse_cli_args(allocator: std.mem.Allocator) !CLIArgs {
    // First we specify what parameters our program can take.
    // We can use `parseParamsComptime` to parse a string into an array of `Param(Help)`
    const params = comptime clap.parseParamsComptime(
        \\-h, --help                Display this help and exit.
        \\-p, --phases <usize>      Number of interleaved phases.
        \\-b, --buffer_size <usize> Maximum buffer size, the output buffer size is adapted accordingly.
        \\-v, --verbose             Print more info.
        \\-f, --file <str>          Use file as input instead of stdin. Raw i32 data type is assumed.
        \\<usize>                   The number of samples per symbol.
        \\<usize>...                Demodulated frequency bins. At the moment only the last value is used.
        \\
    );

    // Initialize our diagnostics, which can be used for reporting useful errors.
    // This is optional. You can also pass `.{}` to `clap.parse` if you don't
    // care about the extra information `Diagnostics` provides.
    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, clap.parsers.default, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        // Report useful error and exit
        diag.report(std.io.getStdErr().writer(), err) catch {};
        return err;
    };
    defer res.deinit();
    if (res.args.help != 0) {
        try clap.help(std.io.getStdErr().writer(), clap.Help, &params, .{});
        return CLIArgs.init_help_only();
    } else if (res.positionals.len < 2) {
        std.debug.print("Need symbol size and at least one demodulation frequency bin.\n", .{});
        return InputError.NotEnoughArguments;
    }
    const phases = res.args.phases orelse 1;
    const symbol_size: usize = res.positionals[0];
    if (symbol_size % phases != 0) {
        return InputError.PhasesValueInvalid;
    }
    const buffer_size = res.args.buffer_size orelse symbol_size;
    if (buffer_size < symbol_size)
        return InputError.BufferSizeTooSmall;

    return CLIArgs.init(allocator, phases, symbol_size, res.positionals[1..], buffer_size, res.args.help != 0, res.args.file);
}

fn debug_print(comptime fmt: []const u8, args: anytype) void {
    const DEBUG_PRINT = true;
    if (DEBUG_PRINT) {
        std.debug.print(fmt, args);
        std.debug.print("\n", .{});
    }
}
