import unittest
from neokernel.calibration import build_calibration, equivalent_estimates
from neokernel.judge import OFFICIAL


class CalibrationTests(unittest.TestCase):
    def test_accepted_factor_refreshes_from_new_native(self):
        import tempfile
        from pathlib import Path
        from neokernel.calibration import refresh_host_factors
        from neokernel.storage import write_json
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_json(directory/'calibration.json', {'accepted': True, 'workloads': {}})
            native = dict(OFFICIAL['public-0'], tps=28.5, source='modal-container-native',
                          sample_count=1, correctness={'passed': True})
            candidate = dict(name='public-0', tps=40, ttft_median=.02, tpot_median=.02)
            run = dict(gpu_tier='H100', native={'public-0': native}, workloads=[candidate], eligible=True)
            result = refresh_host_factors(run, directory)
            self.assertFalse(result['within_15_percent'])
            self.assertEqual(equivalent_estimates(candidate, result)['tps'], 80)
            native['tps'] = 57
            result = refresh_host_factors(run, directory)
            self.assertEqual(equivalent_estimates(candidate, result)['tps'], 40)

    def test_ratios_and_estimates_do_not_mutate_measurements(self):
        workloads = [dict(name=name, tps=ref["tps"]*1.1, ttft_median=ref["ttft_median"]*1.05,
                          tpot_median=ref["tpot_median"]*.95, samples=[{}, {}, {}], correctness={"passed": True},
                          source="modal-container-native", sample_count=3)
                     for name, ref in OFFICIAL.items()]
        native = {w["name"]: w for w in workloads}
        record = build_calibration({"gpu_tier": "H100", "workloads": workloads, "native": native, "eligible": True})
        self.assertTrue(record["within_15_percent"])
        self.assertAlmostEqual(equivalent_estimates(workloads[0], record)["tps"], 57)
        self.assertAlmostEqual(workloads[0]["tps"], 62.7)
        workloads[0]["tps"] = 57*1.2
        invalid = build_calibration({"gpu_tier": "H100", "workloads": workloads, "native": native, "eligible": True})
        self.assertFalse(invalid["within_15_percent"])
        self.assertIsNone(equivalent_estimates(workloads[0], invalid))

    def test_l4_cannot_calibrate_h100(self):
        with self.assertRaises(ValueError):
            build_calibration({"gpu_tier": "L4"})

    def test_host_factor_uses_native_not_candidate(self):
        native = {name: dict(ref, source='modal-container-native', sample_count=3,
                             correctness={'passed': True}) for name, ref in OFFICIAL.items()}
        candidates = [dict(name=name, tps=ref['tps']*2) for name, ref in OFFICIAL.items()]
        record = build_calibration(dict(gpu_tier='H100', native=native, workloads=candidates, eligible=True))
        self.assertTrue(record['within_15_percent'])
        self.assertEqual(record['workloads']['public-0']['local_to_official_ratio']['tps'], 1)
