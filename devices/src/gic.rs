// Copyright 2020, ARM Limited.
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

use std::result;
use std::sync::{Arc, Mutex};

use anyhow::anyhow;
use arch::layout;
use hypervisor::arch::aarch64::gic::{GicState, RedistRegion, Vgic, VgicConfig};
use vm_device::interrupt::{
    InterruptIndex, InterruptManager, InterruptSourceConfig, InterruptSourceGroup,
    LegacyIrqSourceConfig, MsiIrqGroupConfig,
};
use vm_memory::address::Address;
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};
use vmm_sys_util::eventfd::EventFd;

use super::interrupt_controller::{Error, InterruptController};

type Result<T> = result::Result<T, Error>;

// Reserve 32 IRQs for legacy devices.
pub const IRQ_LEGACY_BASE: usize = layout::IRQ_BASE as usize;
pub const IRQ_LEGACY_COUNT: usize = 32;
pub const GIC_SNAPSHOT_ID: &str = "gic-v3-its";

// Gic (Generic Interrupt Controller) struct provides all the functionality of a
// GIC device. It wraps a hypervisor-emulated GIC device (Vgic) provided by the
// `hypervisor` crate.
// Gic struct also implements InterruptController to provide interrupt delivery
// service.
pub struct Gic {
    interrupt_source_group: Arc<dyn InterruptSourceGroup>,
    // The hypervisor agnostic virtual GIC
    vgic: Arc<Mutex<dyn Vgic>>,
}

impl Gic {
    #[expect(clippy::needless_pass_by_value)]
    pub fn new(
        vcpu_count: u32,
        redist_counts: &[u64],
        interrupt_manager: Arc<dyn InterruptManager<GroupConfig = MsiIrqGroupConfig>>,
        vm: Arc<dyn hypervisor::Vm>,
    ) -> Result<Gic> {
        let interrupt_source_group = interrupt_manager
            .create_group(MsiIrqGroupConfig {
                base: IRQ_LEGACY_BASE as InterruptIndex,
                count: IRQ_LEGACY_COUNT as InterruptIndex,
            })
            .map_err(Error::CreateInterruptSourceGroup)?;

        let config = Gic::create_config(vcpu_count as u64, redist_counts);
        let vgic = vm.create_vgic(&config).map_err(Error::CreateGic)?;

        let gic = Gic {
            interrupt_source_group,
            vgic,
        };
        gic.enable()?;

        Ok(gic)
    }

    pub fn restore_vgic(&self, state: Option<GicState>) -> Result<()> {
        let mut vgic = self.vgic.lock().unwrap();
        vgic.set_state(&state.unwrap()).map_err(Error::RestoreGic)
    }

    fn enable(&self) -> Result<()> {
        // Set irqfd for legacy interrupts
        self.interrupt_source_group
            .enable()
            .map_err(Error::EnableInterrupt)?;

        // Set irq_routing for legacy interrupts.
        //   irqchip: Hardcode to 0 as we support only 1 GIC
        //   pin: Use irq number as pin
        for i in IRQ_LEGACY_BASE..(IRQ_LEGACY_BASE + IRQ_LEGACY_COUNT) {
            let config = LegacyIrqSourceConfig {
                irqchip: 0,
                pin: (i - IRQ_LEGACY_BASE) as u32,
            };
            self.interrupt_source_group
                .update(
                    i as InterruptIndex,
                    InterruptSourceConfig::LegacyIrq(config),
                    false,
                    false,
                )
                .map_err(Error::EnableInterrupt)?;
        }

        self.interrupt_source_group
            .set_gsi()
            .map_err(Error::EnableInterrupt)?;
        Ok(())
    }

    /// Default config implied by arch::layout
    pub fn create_default_config(vcpu_count: u64) -> VgicConfig {
        Self::create_config(vcpu_count, &[vcpu_count])
    }

    /// Create a VgicConfig with one or more redistributor regions placed downward
    /// below `layout::GIC_V3_DIST_START`, and the ITS placed below the lowest
    /// redistributor region.
    pub fn create_config(vcpu_count: u64, redist_counts: &[u64]) -> VgicConfig {
        let counts: Vec<u64> = {
            let mut c: Vec<u64> = redist_counts.iter().copied().filter(|&n| n > 0).collect();
            let sum: u64 = c.iter().sum();
            if c.is_empty() || sum > vcpu_count {
                vec![vcpu_count]
            } else if sum < vcpu_count {
                c.push(vcpu_count - sum);
                c
            } else {
                c
            }
        };

        let mut current_addr = layout::GIC_V3_DIST_START.raw_value();
        let mut redist_regions = Vec::with_capacity(counts.len());
        for count in counts {
            let size = layout::GIC_V3_REDIST_SIZE * count;
            current_addr -= size;
            redist_regions.push(RedistRegion {
                base_addr: current_addr,
                size,
                count,
            });
        }

        VgicConfig {
            vcpu_count,
            dist_addr: layout::GIC_V3_DIST_START.raw_value(),
            dist_size: layout::GIC_V3_DIST_SIZE,
            redist_regions,
            msi_addr: current_addr - layout::GIC_V3_ITS_SIZE,
            msi_size: layout::GIC_V3_ITS_SIZE,
            nr_irqs: layout::IRQ_NUM,
        }
    }

    /// Resolve a given `vcpu_id`'s `gicr_base_address` from the configured `redist_regions`.
    pub fn vcpu_gicr_base_addr(redist_regions: &[RedistRegion], vcpu_id: u64) -> u64 {
        let mut remaining = vcpu_id;
        for region in redist_regions {
            if remaining < region.count {
                return region.base_addr + remaining * layout::GIC_V3_REDIST_SIZE;
            }
            remaining -= region.count;
        }
        panic!("vcpu_id {vcpu_id} out of range for redist_regions {redist_regions:?}");
    }

    pub fn get_vgic(&mut self) -> Result<Arc<Mutex<dyn Vgic>>> {
        Ok(Arc::clone(&self.vgic))
    }
}

impl InterruptController for Gic {
    // This should be called anytime an interrupt needs to be injected into the
    // running guest.
    fn service_irq(&mut self, irq: usize) -> Result<()> {
        self.interrupt_source_group
            .trigger(irq as InterruptIndex)
            .map_err(Error::TriggerInterrupt)?;

        Ok(())
    }

    fn notifier(&self, irq: usize) -> Option<EventFd> {
        self.interrupt_source_group.notifier(irq as InterruptIndex)
    }
}

impl Snapshottable for Gic {
    fn id(&self) -> String {
        GIC_SNAPSHOT_ID.to_string()
    }

    fn snapshot(&mut self) -> result::Result<Snapshot, MigratableError> {
        let state = self.vgic.lock().unwrap().state().unwrap();
        Snapshot::new_from_state(&state)
    }
}

impl Pausable for Gic {
    fn pause(&mut self) -> result::Result<(), MigratableError> {
        // Flush tables to guest RAM
        self.vgic.lock().unwrap().save_data_tables().map_err(|e| {
            MigratableError::Pause(anyhow!("Could not save GICv3ITS GIC pending tables {e:?}",))
        })?;
        Ok(())
    }
}
impl Transportable for Gic {}
impl Migratable for Gic {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_config_multi_region_and_vcpu_gicr_base_addr() {
        let config = Gic::create_config(4, &[2, 2]);
        assert_eq!(config.redist_regions.len(), 2);

        let dist_start = layout::GIC_V3_DIST_START.raw_value();
        let r0 = &config.redist_regions[0];
        let r1 = &config.redist_regions[1];

        assert_eq!(r0.count, 2);
        assert_eq!(r0.size, 2 * layout::GIC_V3_REDIST_SIZE);
        assert_eq!(r0.base_addr, dist_start - r0.size);

        assert_eq!(r1.count, 2);
        assert_eq!(r1.size, 2 * layout::GIC_V3_REDIST_SIZE);
        assert_eq!(r1.base_addr, r0.base_addr - r1.size);

        assert_eq!(config.msi_addr, r1.base_addr - layout::GIC_V3_ITS_SIZE);

        assert_eq!(
            Gic::vcpu_gicr_base_addr(&config.redist_regions, 0),
            r0.base_addr
        );
        assert_eq!(
            Gic::vcpu_gicr_base_addr(&config.redist_regions, 1),
            r0.base_addr + layout::GIC_V3_REDIST_SIZE
        );
        assert_eq!(
            Gic::vcpu_gicr_base_addr(&config.redist_regions, 2),
            r1.base_addr
        );
        assert_eq!(
            Gic::vcpu_gicr_base_addr(&config.redist_regions, 3),
            r1.base_addr + layout::GIC_V3_REDIST_SIZE
        );
    }
}
