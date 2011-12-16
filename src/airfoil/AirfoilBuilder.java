package airfoil;


import com.maxeler.maxcompiler.v1.managers.MAX3BoardModel;

public class AirfoilBuilder {

	public static void main(String[] args) {
		AifoilManager m = new AifoilManager(MAX3BoardModel.MAX3424A, "AirfoilResCalc", 1);

		m.build();
	}

}
