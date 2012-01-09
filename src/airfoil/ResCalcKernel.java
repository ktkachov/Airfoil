package airfoil;

import static utils.Utils.array2_t;
import static utils.Utils.array4_t;
import static utils.Utils.float_t;

import com.maxeler.maxcompiler.v1.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v1.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v1.kernelcompiler.SMIO;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Count;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Count.Counter;
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

	private final int input_data_count_width = 32;
	private final HWType input_data_count_t = hwUInt(input_data_count_width);
	private final int partition_size = 1<<10;
	private final int halo_size = 1<<7;

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

		HWVar nhd1Size = io.scalarInput("nhd1Size", input_data_count_t);
		HWVar nhd2Size = io.scalarInput("nhd2Size", input_data_count_t);
		HWVar intraHaloSize = io.scalarInput("intraHaloSize", input_data_count_t);
		HWVar haloDataSize = io.scalarInput("halo_size", input_data_count_t);

		SMIO read_from_host_sm = addStateMachine("host_read", new ResInputSM(this, 10));
		HWVar read_from_host = read_from_host_sm.getOutput("output");
		KStruct input_data_dram = io.input("input_dram", input_struct_t);
		KStruct input_data_host = io.input("input_host", input_struct_t, read_from_host);

		HWVar gm1 = io.scalarInput("gm1", float_t);
		HWVar eps = io.scalarInput("eps", float_t);

		Count.Params ram_write_count_params = control.count.makeParams(MathUtils.bitsToAddress(partition_size));
		Counter ram_write_count = control.count.makeCounter(ram_write_count_params);
		RamPortParams<KStruct> ram_params_write = mem.makeRamPortParams(RamPortMode.WRITE_ONLY, ram_write_count.getCount(), input_data_dram.getType())
													.withDataIn(input_data_dram);

		Count.Params ram_read_count_params = control.count.makeParams(MathUtils.bitsToAddress(partition_size));
		Counter ram_read_count = control.count.makeCounter(ram_read_count_params);
		RamPortParams<KStruct> ram_params_read = mem.makeRamPortParams(RamPortMode.READ_ONLY, ram_read_count.getCount(), input_data_dram.getType());

		KStruct ram_output = mem.ramDualPort(partition_size, RamWriteMode.READ_FIRST, ram_params_write, ram_params_read).getOutputB();



		Count.Params host_ram_write_count_params = control.count.makeParams(MathUtils.bitsToAddress(halo_size))
													.withEnable(read_from_host);
		Counter host_ram_write_count = control.count.makeCounter(host_ram_write_count_params);
		RamPortParams<KStruct> host_ram_params_write = mem.makeRamPortParams(RamPortMode.WRITE_ONLY, host_ram_write_count.getCount(), input_data_host.getType())
														.withDataIn(input_data_host)
														.withWriteEnable(read_from_host);
		Count.Params host_ram_read_count_params = control.count.makeParams(MathUtils.bitsToAddress(halo_size));
		Counter host_ram_read_count = control.count.makeCounter(host_ram_read_count_params);
		RamPortParams<KStruct> host_ram_params_read = mem.makeRamPortParams(RamPortMode.READ_ONLY, host_ram_read_count.getCount(), input_data_dram.getType());

		KStruct host_ram_output = mem.ramDualPort(halo_size, RamWriteMode.READ_FIRST, host_ram_params_write, host_ram_params_read).getOutputB();

		KStruct result_dram = doResMath(ram_output, eps, gm1);
		KStruct result_host = doResMath(host_ram_output, eps, gm1);

		io.output("result_dram", result_dram.getType()) <== result_dram;
		io.output("result_host", result_host.getType()) <== result_host;
	}

	KStruct doResMath(KStruct input_data, HWVar eps, HWVar gm1){

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

		HWVar f = 0.5f*(vol1* q1[0] + vol2* q2[0]) + mu*(q1[0]-q2[0]);
		res1[0] <== f;
		res2[0] <== -f;

		f = 0.5f*(vol1* q1[1] + p1*dy + vol2* q2[1] + p2*dy) + mu*(q1[1]-q2[1]);
		res1[1] <== f;
		res2[1] <== -f;

		f = 0.5f*(vol1* q1[2] - p1*dx + vol2* q2[2] - p2*dx) + mu*(q1[2]-q2[2]);
		res1[2] <== f;
		res2[2] <== -f;

		f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(q2[3]+p2)    ) + mu*(q1[3]-q2[3]);
		res1[3] <== f;
		res2[3] <== -f;

		return result;

	}

}
