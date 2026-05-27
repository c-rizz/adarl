#!/bin/bash
set -e

SCRIPT_NAME="set_cpu_governor.sh"
INSTALL_DIR="/usr/local/sbin"
INSTALL_PATH="$INSTALL_DIR/$SCRIPT_NAME"
CRON_FILE="/etc/cron.d/set_cpu_governor"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $EUID -ne 0 ]]; then
    echo "Not root. Copying installer to /tmp and re-running with sudo..."
    cp "$SCRIPT_DIR/$SCRIPT_NAME" /tmp/
    cp "$SCRIPT_DIR/install_set_cpu_governor.sh" /tmp/
    exec sudo bash /tmp/install_set_cpu_governor.sh
fi

echo "Copying $SCRIPT_NAME to $INSTALL_PATH..."
cp "$SCRIPT_DIR/$SCRIPT_NAME" "$INSTALL_PATH"
chmod 755 "$INSTALL_PATH"
chown root:root "$INSTALL_PATH"

echo "Installing cron job to $CRON_FILE..."
cat > "$CRON_FILE" <<EOF
# Run set_cpu_governor at every boot
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
@reboot root $INSTALL_PATH >> /var/log/set_cpu_governor.log 2>&1
EOF
chmod 644 "$CRON_FILE"

echo "Done. $INSTALL_PATH will run at every boot."
echo "You can test it now with: sudo $INSTALL_PATH"
