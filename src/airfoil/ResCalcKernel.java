package airfoil;

import static utils.Utils.array2_t;
import static utils.Utils.array4_t;
import static utils.Utils.float_t;

import com.maxeler.maxcompiler.v1.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v1.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Count;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Count.Counter;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Count.WrapMode;
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

//	private final KStructType.StructFieldType x1_f = KStructType.sft("x1", array2_t);
//	private final KStructType.StructFieldType x2_f = KStructType.sft("x2", array2_t);
//	private final KStructType.StructFieldType q1_f = KStructType.sft("q1", array4_t);
//	private final KStructType.StructFieldType q2_f = KStructType.sft("q2", array4_t);
//	private final KStructType.StructFieldType adt1_f = KStructType.sft("adt1", float_t);
//	private final KStructType.StructFieldType adt2_f = KStructType.sft("adt2", float_t);
	private final KStructType res_struct_t
		= new KStructType(
				KStructType.sft("x1", array2_t),
				KStructType.sft("x2", array2_t),
				KStructType.sft("q1", array4_t),
				KStructType.sft("q2", array4_t),
				KStructType.sft("adt1", float_t),
				KStructType.sft("adt2", float_t)
			);


	public ResCalcKernel(KernelParameters params) {
		super(params);

		HWVar nhd1Size = io.scalarInput("nhd1Size", input_data_count_t);
		HWVar nhd2Size = io.scalarInput("nhd2Size", input_data_count_t);
		HWVar intraHaloSize = io.scalarInput("intraHaloSize", input_data_count_t);

		HWVar haloDataSize = io.scalarInput("halo_size", input_data_count_t);
		Counter host_halo_counter = control.count.makeCounter(control.count.makeParams(4).withMax(10));
		Count.Params count_params = control.count.makeParams(input_data_count_width)
									.withWrapMode(WrapMode.STOP_AT_MAX)
									.withMax(nhd1Size);

		Counter nhd1_in_count = control.count.makeCounter(count_params);

		count_params = control.count.makeParams(input_data_count_width)
						.withMax(nhd2Size)
						.withWrapMode(WrapMode.STOP_AT_MAX)
						.withEnable(nhd1_in_count.getCount() . eq(nhd1Size));

		Counter nhd2_in_count = control.count.makeCounter(count_params);

		HWVar totalSize = nhd1Size + nhd2Size + intraHaloSize;

		HWVar kernel_count = control.count.simpleCounter(48);
		HWVar input_enabled = kernel_count . lte(totalSize.cast(kernel_count.getType()));
//		HWVar nhd1_in_enable = nhd1_in_count.getCount() . neq(nhd1Size);



		KStruct input_data = io.input("input", res_struct_t);

		HWVar gm1 = io.scalarInput("gm1", float_t);
		HWVar eps = io.scalarInput("eps", float_t);

		Count.Params ram_write_count_params = control.count.makeParams(MathUtils.bitsToAddress(partition_size));
		Counter ram_write_count = control.count.makeCounter(ram_write_count_params);
		RamPortParams<KStruct> ram_params_write = mem.makeRamPortParams(RamPortMode.WRITE_ONLY, ram_write_count.getCount(), input_data.getType())
													.withDataIn(input_data);

		Count.Params ram_read_count_params = control.count.makeParams(MathUtils.bitsToAddress(partition_size));
		Counter ram_read_count = control.count.makeCounter(ram_read_count_params);
		RamPortParams<KStruct> ram_params_read = mem.makeRamPortParams(RamPortMode.READ_ONLY, ram_read_count.getCount(), input_data.getType());

		KStruct ram_output = mem.ramDualPort(partition_size, RamWriteMode.READ_FIRST, ram_params_write, ram_params_read).getOutputB();

		KArray<HWVar> x1 = ram_output["x1"];
		KArray<HWVar> x2 = ram_output["x2"];
		KArray<HWVar> q1 = ram_output["q1"];
		KArray<HWVar> q2 = ram_output["q2"];
		HWVar adt1 = ram_output["adt1"];
		HWVar adt2 = ram_output["adt2"];
		HWVar mu = 0.5f*(adt1+adt2)*eps;

		HWVar dx = x1[0] - x2[0];
		HWVar dy = x1[1] - x2[1];
		HWVar ri = 1.0f / q1[0];
		HWVar p1 = gm1 * (q1[3] - 0.5f*ri*( q1[1] * q1[1] + q1[2] * q1[2]) );
		HWVar vol1 = ri * (q1[1]*dy - q1[2]*dx);

		ri = 1.0f / q1[0];
		HWVar p2 = gm1*(q2[3]-0.5f*ri*(q2[1]*q2[1]+q2[2]*q2[2]));
		HWVar vol2 = ri*(q2[1]*dy - q2[2]*dx);

		KArray<HWVar> res1 = array4_t.newInstance(this);
		KArray<HWVar> res2 = array4_t.newInstance(this);

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

		io.output("res1", res1.getType()) <== res1;
		io.output("res2", res2.getType()) <== res2;
	}

}
