/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Windows;
using System.Windows.Controls;

namespace CADRobotExporter.UI
{
    /// <summary>
    /// Reports where the exported files were written. A true dialog result means the user asked to
    /// open the containing folder.
    /// </summary>
    public class ExportCompleteDialog : Window
    {
        public ExportCompleteDialog(string urdfPath, string mjcfPath, string superdexBotPath)
        {
            Theme.FluentTheme.Apply(this);

            Title = "SuperDex CAD Exporter";
            Topmost = true;
            Width = 720;
            SizeToContent = SizeToContent.Height;
            ResizeMode = ResizeMode.NoResize;
            WindowStartupLocation = WindowStartupLocation.CenterScreen;
            ShowInTaskbar = false;

            string message = "Export complete!\n\n" +
                "URDF file saved to:\n" + urdfPath + "\n\n" +
                "MJCF file saved to:\n" + mjcfPath + "\n\n" +
                "SuperDex Bot file saved to:\n" + superdexBotPath;

            TextBlock label = new TextBlock
            {
                Text = message,
                TextWrapping = TextWrapping.Wrap,
            };

            Button openFolderButton = new Button
            {
                Content = "Open Folder",
                IsDefault = true,
                Width = 200,
                Height = 32,
            };
            openFolderButton.Click += (sender, e) => { DialogResult = true; };

            Button closeButton = new Button
            {
                Content = "Close",
                IsCancel = true,
                Width = 80,
                Height = 32,
                Margin = new Thickness(20, 0, 0, 0),
            };

            StackPanel buttons = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(0, 20, 0, 0),
            };
            buttons.Children.Add(openFolderButton);
            buttons.Children.Add(closeButton);

            StackPanel root = new StackPanel { Margin = new Thickness(10) };
            root.Children.Add(label);
            root.Children.Add(buttons);

            Content = root;
        }
    }
}
