const std = @import("std");
const clap = @import("zig-clap/clap.zig");
const goertzel = @import("goertzel.zig");
const InputError = error{ NotEnoughArguments, PhasesValueInvalid, BufferSizeTooSmall };

pub fn main() !void {
    // prepare goertzel core
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());

    defer _ = arena.deinit();
    const settings = try parse_cli_args(arena.allocator());
    defer settings.deinit();
    if (settings.help_only) return;
    var core = try goertzel.Core.init(arena.allocator(), settings.symbol_size, settings.phases, settings.k);

    // file input buffer
    var path_buffer: [std.fs.MAX_PATH_BYTES]u8 = undefined;
    const inputfile = if (settings.file) |file| try std.fs.openFileAbsolute(try std.fs.realpath(file, &path_buffer), .{}) else std.io.getStdIn();
    defer inputfile.close();
    const max_buff_size = settings.buffer_size;
    const buffer_size = max_buff_size - (max_buff_size % settings.symbol_size);
    debug_print("Max. buffer size: {}\nActual buffer size: {}", .{ max_buff_size, buffer_size });
    var buffer = try arena.allocator().alloc(i32, buffer_size);
    var bytes_written: usize = 0;

    // file output buffer
    const stdout_file = std.io.getStdOut().writer();
    var writer = std.io.bufferedWriter(stdout_file);
    const samples_per_phase = settings.symbol_size / settings.phases;
    var out_buffer = try arena.allocator().alloc(f64, 2 * settings.k.len * buffer_size / (samples_per_phase));

    // process samples
    while (true) {
        debug_print("pushing buffer", .{});
        const bytes_read = try inputfile.readAll(std.mem.sliceAsBytes(buffer));
        const samples_read = bytes_read / @sizeOf(i32);
        if (samples_read > 0 and samples_read % samples_per_phase == 0) {
            const result_count = try core.push(buffer[0..samples_read], out_buffer);
            bytes_written += try writer.write(std.mem.sliceAsBytes(out_buffer[0 .. 2 * result_count]));
            debug_print("out buffer: {any}", .{out_buffer}); //
        } else {
            break;
        }
    }
    try writer.flush(); // don't forget to flush!
    try std.fmt.format(std.io.getStdErr().writer(), "\nbytes written: {d}\n", .{bytes_written});
}

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

inline fn debug_print(comptime fmt: []const u8, args: anytype) void {
    const DEBUG_PRINT = false;
    if (DEBUG_PRINT) {
        std.debug.print(fmt, args);
        std.debug.print("\n", .{});
    }
}
