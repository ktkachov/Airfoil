package airfoil;

import static utils.Utils.array2_t;
import static utils.Utils.array4_t;
import static utils.Utils.float_t;

import com.maxeler.maxcompiler.v1.kernelcompiler.Kernel;
import com.maxeler.maxcompiler.v1.kernelcompiler.KernelParameters;
import com.maxeler.maxcompiler.v1.kernelcompiler.SMIO;
import com.maxeler.maxcompiler.v1.kernelcompiler.stdlib.core.Mem;
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

	private final int max_partition_size = 1<<13;
	private final int halo_size = 1<<7;

	private final KStructType node_struct_t
		= new KStructType(
				KStructType.sft("x", array2_t)
			);

	private final KStructType cell_struct_t
		= new KStructType(
				KStructType.sft("q", array4_t),
				KStructType.sft("adt", float_t)
		);

	final int addr_width = MathUtils.bitsToAddress(max_partition_size);
	final HWType addr_t = hwUInt(addr_width);
	private final KStructType address_struct_t
		= new KStructType(
				KStructType.sft("node1", addr_t),
				KStructType.sft("node2", addr_t),
				KStructType.sft("cell1", addr_t),
				KStructType.sft("cell2", addr_t)
		);

	private final KStructType res_struct_t
		= new KStructType(
				KStructType.sft("res1", array4_t),
				KStructType.sft("res2", array4_t)
			);


	public ResCalcKernel(KernelParameters params) {
		super(params);

		HWVar nhd1Size = io.scalarInput("nhd1Size", addr_t);
		HWVar nhd2Size = io.scalarInput("nhd2Size", addr_t);
		HWVar intraHaloSize = io.scalarInput("intraHaloSize", addr_t);
		HWVar haloDataSize = io.scalarInput("halo_size", addr_t);

		HWVar partition_size = nhd1Size + nhd2Size + intraHaloSize;


		SMIO control_sm = addStateMachine("io_control_sm", new ResControlSM(this, addr_width, 10));
		control_sm.connectInput("host_halo_size", haloDataSize);
		control_sm.connectInput("nhd1_size", nhd1Size);
		control_sm.connectInput("nhd2_size", nhd2Size);
		control_sm.connectInput("intra_halo_size", intraHaloSize);


		HWVar reading_data = control_sm.getOutput("reading");
		HWVar processing_data = control_sm.getOutput("processing");
		HWVar outputting_data = control_sm.getOutput("writing");
		HWVar read_host_halo = control_sm.getOutput("halo_read");

		KStruct node_input_dram = io.input("node_input_dram", node_struct_t, reading_data);
		KStruct cell_input_dram = io.input("cell_input_dram", cell_struct_t, reading_data);
		KStruct address_struct = io.input("addresses", address_struct_t);
		KStruct cell_data_host = io.input("input_host", cell_struct_t, read_host_halo);

		HWVar gm1 = io.scalarInput("gm1", float_t);
		HWVar eps = io.scalarInput("eps", float_t);


		HWVar ram_write_counter = control.count.simpleCounter(addr_width, max_partition_size);

		// RAMs for the node data
		Mem.RamPortParams<KStruct> node_ram_portA_params
			= mem.makeRamPortParams(RamPortMode.READ_WRITE, ram_write_counter, node_struct_t)
				.withDataIn(node_input_dram)
				.withWriteEnable(reading_data)
				;
		HWVar node1_addr = address_struct["node1"];
		RamPortParams<KStruct> node_ram1_portB_params
			= mem.makeRamPortParams(RamPortMode.READ_ONLY, node1_addr, node_struct_t);
		Mem.DualPortMemOutputs<KStruct> node_ram1_output
			= mem.ramDualPort(max_partition_size, RamWriteMode.WRITE_FIRST, node_ram_portA_params, node_ram1_portB_params);


		HWVar node2_addr = address_struct["node2"];
		RamPortParams<KStruct> node_ram2_portB_params
			= mem.makeRamPortParams(RamPortMode.READ_ONLY, node2_addr, node_struct_t);
		Mem.DualPortMemOutputs<KStruct> node_ram2_output
			= mem.ramDualPort(max_partition_size, RamWriteMode.WRITE_FIRST, node_ram_portA_params, node_ram2_portB_params);

		//RAMs for cell data
		Mem.RamPortParams<KStruct> cell_ram_portA_params
		= mem.makeRamPortParams(RamPortMode.READ_WRITE, ram_write_counter, cell_struct_t)
			.withDataIn(cell_input_dram)
			.withWriteEnable(reading_data)
			;
		HWVar cell1_addr = address_struct["cell1"];
		RamPortParams<KStruct> cell_ram1_portB_params
			= mem.makeRamPortParams(RamPortMode.READ_ONLY, cell1_addr, cell_struct_t);
		Mem.DualPortMemOutputs<KStruct> cell_ram1_output
			= mem.ramDualPort(max_partition_size, RamWriteMode.WRITE_FIRST, cell_ram_portA_params, cell_ram1_portB_params);

		HWVar cell2_addr = address_struct["cell2"];
		RamPortParams<KStruct> cell_ram2_portB_params
			= mem.makeRamPortParams(RamPortMode.READ_ONLY, cell2_addr, cell_struct_t);
		Mem.DualPortMemOutputs<KStruct> cell_ram2_output
			= mem.ramDualPort(max_partition_size, RamWriteMode.WRITE_FIRST, cell_ram_portA_params, cell_ram2_portB_params);


		//The arithmetic pipeline
		KStruct res_vector = doResMath(
					node_ram1_output.getOutputB(),
					node_ram2_output.getOutputB(),
					cell_ram1_output.getOutputB(),
					cell_ram2_output.getOutputB(),
					eps,
					gm1
				);



		KArray<HWVar> previous_res_value_cell1 = array4_t.newInstance(this);
		KArray<HWVar> previous_res_value_cell2 = array4_t.newInstance(this);

		KArray<HWVar> new_res_value_cell1 = array4_t.newInstance(this);
		KArray<HWVar> new_res_value_cell2 = array4_t.newInstance(this);
		KArray<HWVar> zeroes = array4_t.newInstance(this);

		for (int i = 0; i < array4_t.getSize(); ++i) {
			KArray<HWVar> res1 = res_vector.get("res1");
			new_res_value_cell1[i] <== previous_res_value_cell1[i] + res1[i];

			KArray<HWVar> res2 = res_vector.get("res2");
			new_res_value_cell2[i] <== previous_res_value_cell2[i] + res2[i];

			zeroes[i] <== float_t.newInstance(this, 0.0);

		}


		//Rams for res data
		Mem.RamPortParams<KArray<HWVar>> res_ram1_portA_params
			= mem.makeRamPortParams(RamPortMode.READ_WRITE, cell1_addr, array4_t)
				.withDataIn(new_res_value_cell1)
			;
		Mem.RamPortParams<KArray<HWVar>> res_ram1_portB_params
			= mem.makeRamPortParams(RamPortMode.READ_WRITE, ram_write_counter, array4_t)
				.withDataIn(zeroes)
				.withWriteEnable(hwBool().newInstance(this, false))
				;
		DualPortMemOutputs<KArray<HWVar>> res_ram1_output = mem.ramDualPort(max_partition_size, RamWriteMode.READ_FIRST, res_ram1_portA_params, res_ram1_portB_params);

		previous_res_value_cell1 <== stream.offset(res_ram1_output.getOutputA(), -14);


		Mem.RamPortParams<KArray<HWVar>> res_ram2_portA_params
		= mem.makeRamPortParams(RamPortMode.READ_WRITE, cell2_addr, array4_t)
			.withDataIn(new_res_value_cell2)
		;
		Mem.RamPortParams<KArray<HWVar>> res_ram2_portB_params
			= mem.makeRamPortParams(RamPortMode.READ_WRITE, ram_write_counter, array4_t)
				.withDataIn(zeroes)
				.withWriteEnable(hwBool().newInstance(this, false))
				;
		DualPortMemOutputs<KArray<HWVar>> res_ram2_output = mem.ramDualPort(max_partition_size, RamWriteMode.READ_FIRST, res_ram2_portA_params, res_ram2_portB_params);

		previous_res_value_cell2 <== stream.offset(res_ram2_output.getOutputA(), -14);



		KArray<HWVar> res_output = array4_t.newInstance(this);
		for (int i = 0; i < res_output.getSize(); ++i) {
			res_output[i] <== res_ram1_output.getOutputB()[i] + res_ram2_output.getOutputB()[i];
		}


		io.output("result_dram", res_output.getType()) <== res_output;
	}


	// The math that produces the res1 and res2 vectors
	KStruct doResMath(KStruct node1, KStruct node2, KStruct cell1, KStruct cell2, HWVar eps, HWVar gm1){

		KArray<HWVar> x1 = node1["x"];
		KArray<HWVar> x2 = node2["x"];
		KArray<HWVar> q1 = cell1["q"];
		KArray<HWVar> q2 = cell2["q"];
		HWVar adt1 = cell1["adt"];
		HWVar adt2 = cell2["adt"];
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
		res1[0] <==  f;
		res2[0] <== -f;

		f = 0.5f*(vol1* q1[1] + p1*dy + vol2* q2[1] + p2*dy) + mu*(q1[1]-q2[1]);
		res1[1] <==  f;
		res2[1] <== -f;

		f = 0.5f*(vol1* q1[2] - p1*dx + vol2* q2[2] - p2*dx) + mu*(q1[2]-q2[2]);
		res1[2] <==  f;
		res2[2] <== -f;

		f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(q2[3]+p2)    ) + mu*(q1[3]-q2[3]);
		res1[3] <==  f;
		res2[3] <== -f;

		return result;

	}

}
