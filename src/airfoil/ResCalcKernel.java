package airfoil;

import static utils.Utils.array2_t;
import static utils.Utils.array4_t;
import static utils.Utils.float_t;

import com.maxeler.maxcompiler.v1.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v1.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v1.kernelcompiler.SMIO;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Count;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Count.WrapMode;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Mem.DualPortMemOutputs;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Mem.RamPortMode;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Mem.RamPortParams;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Mem.RamWriteMode;
import com.maxeler.maxcompiler.v1.kernelcompiler.types.base.HWType;
import com.maxeler.maxcompiler.v1.kernelcompiler.types.base.HWVar;
import com.maxeler.maxcompiler.v1.kernelcompiler.types.composite.KArray;
import com.maxeler.maxcompiler.v1.kernelcompiler.types.composite.KStruct;
import com.maxeler.maxcompiler.v1.kernelcompiler.types.composite.KStructType;
import com.maxeler.maxcompiler.v1.utils.MathUtils;


public class ResCalcKernel extends Kernel {

	private final int max_partition_size = 1<<14;
	private final int halo_size = 1<<7;
	private final int ram_size = halo_size + max_partition_size;
	private final int input_data_count_width = MathUtils.bitsToAddress(ram_size);
	private final HWType input_data_count_t = hwUInt(input_data_count_width);

	private final KStructType input_struct_t
		= new KStructType(
				KStructType.sft("x1", array2_t),
				KStructType.sft("x2", array2_t),
				KStructType.sft("q1", array4_t),
				KStructType.sft("q2", array4_t),
				KStructType.sft("adt1", float_t),
				KStructType.sft("adt2", float_t)
			);

	private final KStructType res_struct_t
		= new KStructType(
				KStructType.sft("res1", array4_t),
				KStructType.sft("res2", array4_t)
			);


	public ResCalcKernel(KernelParameters params) {
		super(params);

		final int addr_width = MathUtils.bitsToAddress(ram_size);

		HWVar nhd1Size = io.scalarInput("nhd1Size", input_data_count_t);
		HWVar nhd2Size = io.scalarInput("nhd2Size", input_data_count_t);
		HWVar intraHaloSize = io.scalarInput("intraHaloSize", input_data_count_t);
		HWVar haloDataSize = io.scalarInput("halo_size", input_data_count_t);

		HWVar partition_size = nhd1Size + nhd2Size + intraHaloSize;

		Count.Params input_counter_params = control.count.makeParams(addr_width)
			.withMax(partition_size)
			.withWrapMode(WrapMode.STOP_AT_MAX);

		HWVar input_count = control.count.makeCounter(input_counter_params).getCount();

		HWVar address = io.input("address", hwUInt(addr_width));

		SMIO control_sm = addStateMachine("io_control_sm", new ResControlSM(this, addr_width, 10));
		control_sm.connectInput("host_halo_size", haloDataSize);
		control_sm.connectInput("nhd1_size", nhd1Size);
		control_sm.connectInput("nhd2_size", nhd2Size);
		control_sm.connectInput("intra_halo_size", intraHaloSize);


		HWVar reading_data = control_sm.getOutput("reading");
		HWVar processing_data = control_sm.getOutput("processing");
		HWVar outputting_data = control_sm.getOutput("writing");
		HWVar read_host_halo = control_sm.getOutput("halo_read");
		KStruct input_data_dram = io.input("input_dram", input_struct_t, reading_data);
		KStruct input_data_host = io.input("input_host", input_struct_t, read_host_halo);

		HWVar gm1 = io.scalarInput("gm1", float_t);
		HWVar eps = io.scalarInput("eps", float_t);


		HWVar halo_write_count
			= control.count.makeCounter(control.count.makeParams(addr_width)
				.withEnable(read_host_halo)
				.withMax(haloDataSize)).getCount();

		RamPortParams<KStruct> ram_params_read
			= mem.makeRamPortParams(RamPortMode.READ_WRITE, read_host_halo ? (halo_write_count + max_partition_size) : address, input_data_dram.getType())
				.withDataIn(input_data_host)
				.withWriteEnable(read_host_halo);


		RamPortParams<KStruct> ram_params_write = mem.makeRamPortParams(RamPortMode.READ_WRITE, input_count, input_data_dram.getType())
			.withDataIn(input_data_dram)
			.withWriteEnable(reading_data);

		KStruct partition_data = mem.ramDualPort(ram_size, RamWriteMode.READ_FIRST, ram_params_write, ram_params_read).getOutputB();


		KStruct res_ram_contents = res_struct_t.newInstance(this);
		KStruct partition_result = doResMath(partition_data, eps, gm1, res_ram_contents);
//		KStruct result_host = doResMath(host_ram_output, eps, gm1, res_ram_contents);


		RamPortParams<KStruct> write_res_params
			= mem.makeRamPortParams(RamPortMode.READ_WRITE, address.cast(hwUInt(addr_width)), partition_result.getType())
				.withDataIn(partition_result)
				.withWriteEnable(processing_data);


		Count.Params output_counter_params = control.count.makeParams(addr_width)
			.withEnable(outputting_data)
			.withWrapMode(WrapMode.COUNT_LT_MAX_THEN_WRAP);
		HWVar output_address = control.count.makeCounter(output_counter_params).getCount();

		RamPortParams<KStruct> read_res_params
			= mem.makeRamPortParams(RamPortMode.READ_ONLY, output_address, partition_result.getType());

		DualPortMemOutputs<KStruct> res_ram_output = mem.ramDualPort(ram_size, RamWriteMode.READ_FIRST, write_res_params, read_res_params);
		res_ram_contents <== stream.offset(res_ram_output.getOutputA(), -14); //FIXME: EXPLAIN!!!

		io.output("result_dram", res_ram_contents.getType(), outputting_data) <== res_ram_output.getOutputB();
//		io.output("result_host", result_host.getType()) <== result_host;
	}


	// The math that produces the res1 and res2 vectors
	KStruct doResMath(KStruct input_data, HWVar eps, HWVar gm1, KStruct current_res){

		KArray<HWVar> x1 = input_data["x1"];
		KArray<HWVar> x2 = input_data["x2"];
		KArray<HWVar> q1 = input_data["q1"];
		KArray<HWVar> q2 = input_data["q2"];
		HWVar adt1 = input_data["adt1"];
		HWVar adt2 = input_data["adt2"];
		HWVar mu = 0.5f*(adt1+adt2)*eps;

		HWVar dx = x1[0] - x2[0];
		HWVar dy = x1[1] - x2[1];
		HWVar ri = 1.0f / q1[0];
		HWVar p1 = gm1 * (q1[3] - 0.5f*ri*( q1[1] * q1[1] + q1[2] * q1[2]) );
		HWVar vol1 = ri * (q1[1]*dy - q1[2]*dx);

		ri = 1.0f / q1[0];
		HWVar p2 = gm1*(q2[3]-0.5f*ri*(q2[1]*q2[1]+q2[2]*q2[2]));
		HWVar vol2 = ri*(q2[1]*dy - q2[2]*dx);

		KStruct result = res_struct_t.newInstance(this);
		KArray<HWVar> res1 = result["res1"];
		KArray<HWVar> res2 = result["res2"];
		KArray<HWVar> curr_res1 = current_res["res1"];
		KArray<HWVar> curr_res2 = current_res["res2"];

		HWVar f = 0.5f*(vol1* q1[0] + vol2* q2[0]) + mu*(q1[0]-q2[0]);
		res1[0] <== curr_res1[0] + f;
		res2[0] <== curr_res2[0] - f;

		f = 0.5f*(vol1* q1[1] + p1*dy + vol2* q2[1] + p2*dy) + mu*(q1[1]-q2[1]);
		res1[1] <== curr_res1[1] + f;
		res2[1] <== curr_res2[1] - f;

		f = 0.5f*(vol1* q1[2] - p1*dx + vol2* q2[2] - p2*dx) + mu*(q1[2]-q2[2]);
		res1[2] <== curr_res1[2] + f;
		res2[2] <== curr_res2[2] - f;

		f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(q2[3]+p2)    ) + mu*(q1[3]-q2[3]);
		res1[3] <== curr_res1[3] + f;
		res2[3] <== curr_res2[3] - f;

		return result;

	}

}
